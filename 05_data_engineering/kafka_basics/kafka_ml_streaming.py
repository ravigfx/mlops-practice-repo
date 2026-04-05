"""
05 - Data Engineering: Apache Kafka — ML Event Streaming
Practice: producer, consumer, real-time prediction logging
Requirements: pip install kafka-python
Local Kafka:  docker-compose up kafka
"""

import json
import time
import uuid
import logging
from datetime import datetime
from typing import Optional

# pip install kafka-python
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("kafka-python not installed. Install: pip install kafka-python")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_PREDICTIONS = "ml.predictions"
TOPIC_FEEDBACK    = "ml.feedback"
TOPIC_ALERTS      = "ml.alerts"


# ══════════════════════════════════════════════════════════════
# 1. Prediction Event Producer
#    Publishes model predictions to Kafka for downstream consumers
# ══════════════════════════════════════════════════════════════

class PredictionProducer:
    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP):
        if not KAFKA_AVAILABLE:
            raise RuntimeError("kafka-python not installed")
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",           # wait for all replicas
            retries=3,
            linger_ms=10,         # batch small messages
        )

    def publish_prediction(self, features: dict, prediction: int,
                           probabilities: list[float], model_version: str,
                           request_id: Optional[str] = None) -> str:
        request_id = request_id or str(uuid.uuid4())
        event = {
            "event_type":    "prediction",
            "request_id":    request_id,
            "timestamp":     datetime.utcnow().isoformat(),
            "model_version": model_version,
            "features":      features,
            "prediction":    prediction,
            "probabilities": probabilities,
            "max_prob":      max(probabilities),
        }
        future = self.producer.send(
            TOPIC_PREDICTIONS,
            key=request_id,
            value=event,
        )
        metadata = future.get(timeout=10)
        log.info(f"Published prediction {request_id} → partition {metadata.partition}")
        return request_id

    def publish_alert(self, alert_type: str, details: dict):
        event = {
            "event_type": "alert",
            "alert_type": alert_type,
            "timestamp":  datetime.utcnow().isoformat(),
            "details":    details,
        }
        self.producer.send(TOPIC_ALERTS, key=alert_type, value=event)
        log.warning(f"Alert published: {alert_type}")

    def close(self):
        self.producer.flush()
        self.producer.close()


# ══════════════════════════════════════════════════════════════
# 2. Prediction Consumer
#    Reads prediction events for monitoring / retraining
# ══════════════════════════════════════════════════════════════

class PredictionConsumer:
    def __init__(self, group_id: str, bootstrap_servers: str = KAFKA_BOOTSTRAP):
        if not KAFKA_AVAILABLE:
            raise RuntimeError("kafka-python not installed")
        self.consumer = KafkaConsumer(
            TOPIC_PREDICTIONS,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            max_poll_records=100,
        )
        self.predictions: list[dict] = []
        self.low_confidence_count = 0

    def consume(self, timeout_ms: int = 5000, max_messages: int = 100):
        """Consume prediction events and run basic monitoring."""
        log.info(f"Consuming from {TOPIC_PREDICTIONS}...")
        msg_count = 0

        for msg in self.consumer:
            event = msg.value
            self.predictions.append(event)
            msg_count += 1

            # Real-time monitoring
            if event.get("max_prob", 1.0) < 0.7:
                self.low_confidence_count += 1
                log.warning(f"Low confidence prediction: {event['request_id']} "
                            f"prob={event['max_prob']:.3f}")

            if msg_count >= max_messages:
                break

        self._summarize()
        return self.predictions

    def _summarize(self):
        if not self.predictions:
            return
        preds = [p["prediction"] for p in self.predictions]
        from collections import Counter
        dist = Counter(preds)
        print(f"\nConsumed {len(self.predictions)} predictions:")
        print(f"  Prediction distribution: {dict(dist)}")
        print(f"  Low confidence alerts: {self.low_confidence_count}")
        avg_conf = sum(p["max_prob"] for p in self.predictions) / len(self.predictions)
        print(f"  Avg max probability: {avg_conf:.3f}")

    def close(self):
        self.consumer.close()


# ══════════════════════════════════════════════════════════════
# 3. Docker Compose for local Kafka
# ══════════════════════════════════════════════════════════════

DOCKER_COMPOSE_KAFKA = """
# docker-compose-kafka.yml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on: [zookeeper]
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    depends_on: [kafka]
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
"""


# ══════════════════════════════════════════════════════════════
# 4. Demo (no Kafka required — simulates events)
# ══════════════════════════════════════════════════════════════

def simulate_kafka_workflow():
    """Demonstrate the event structure without a running Kafka broker."""
    print("\n── Kafka ML Event Simulation ───────────────────────────")
    import random

    # Simulate prediction events
    events = []
    for i in range(10):
        probs = [random.random() for _ in range(3)]
        total = sum(probs)
        probs = [p / total for p in probs]
        pred  = probs.index(max(probs))

        event = {
            "event_type":    "prediction",
            "request_id":    str(uuid.uuid4()),
            "timestamp":     datetime.utcnow().isoformat(),
            "model_version": "1.0.0",
            "features":      {"sepal_length": 5.1, "petal_length": 1.4},
            "prediction":    pred,
            "probabilities": probs,
            "max_prob":      max(probs),
        }
        events.append(event)
        status = "⚠️" if event["max_prob"] < 0.6 else "✅"
        print(f"  {status} Event {i+1}: pred={pred}  conf={event['max_prob']:.3f}")

    # Stats
    from collections import Counter
    dist = Counter(e["prediction"] for e in events)
    low_conf = sum(1 for e in events if e["max_prob"] < 0.6)
    print(f"\nSummary: dist={dict(dist)}  low_confidence={low_conf}/{len(events)}")

    # Save docker-compose file
    with open("docker-compose-kafka.yml", "w") as f:
        f.write(DOCKER_COMPOSE_KAFKA)
    print("\ndocker-compose-kafka.yml saved.")
    print("Start Kafka: docker-compose -f docker-compose-kafka.yml up -d")


if __name__ == "__main__":
    simulate_kafka_workflow()
