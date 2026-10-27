# flake8: noqa
"""Integration tests for metrics functionality."""

import logging
import time
from io import StringIO

import pytest
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from pruna.telemetry import increment_counter, set_telemetry_metrics, track_usage
from pruna.telemetry.metrics import reader, set_opentelemetry_log_level

UNREACHABLE_ENDPOINT = "http://nonexistent-domain-for-testing.invalid/v1/metrics"


def export_metric_to_unreachable_endpoint() -> None:
    """
    Export a single metric to an unreachable endpoint to force an OTLP export error.

    A dedicated exporter is used instead of patching the private internals of pruna's exporter,
    which change between opentelemetry releases. The short timeout makes the exporter give up
    (and log its ERROR) within ``force_flush`` instead of after several retries.
    """
    test_exporter = OTLPMetricExporter(endpoint=UNREACHABLE_ENDPOINT, timeout=1)
    test_reader = PeriodicExportingMetricReader(test_exporter, export_interval_millis=60000)
    test_provider = MeterProvider(metric_readers=[test_reader])
    try:
        test_provider.get_meter("pruna-test").create_counter("test_operation").add(1)
        test_reader.force_flush()
    finally:
        test_provider.shutdown()


@pytest.mark.integration
def test_otlp_export_to_collector():
    """FOR THIS TO RUN YOU WILL NEED:
    Docker compose:
    version: "3"
    services:
      otel-collector:
        image: otel/opentelemetry-collector:latest
        command: ["--config=/etc/otel-collector-config.yaml"]
        volumes:
          - /Users/gabrieltregoat/workspace/prunatree/otel-collector-config.yaml:/etc/otel-collector-config.yaml
        ports:
          - "4318:4318"  # HTTP receiver
          - "8889:8889"  # Prometheus exporter
        networks:
          - monitoring

      prometheus:
        image: prom/prometheus:latest
        ports:
          - "9090:9090"
        volumes:
          - ./prometheus.yml:/etc/prometheus/prometheus.yml
        command:
          - '--config.file=/etc/prometheus/prometheus.yml'
        networks:
          - monitoring
        depends_on:
          - otel-collector

      grafana:
        image: grafana/grafana:latest
        ports:
          - "3000:3000"
        environment:
          - GF_SECURITY_ADMIN_PASSWORD=admin
          - GF_SECURITY_ADMIN_USER=admin
          - GF_AUTH_ANONYMOUS_ENABLED=true
          - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
        volumes:
          - grafana-storage:/var/lib/grafana
        networks:
          - monitoring
        depends_on:
          - prometheus

    networks:
      monitoring:

    volumes:
      grafana-storage:

    That docker compose needs to be running.

    The otel-collector-config.yaml file (update the path in the docker compose):
    receivers:
      otlp:
        protocols:
          http:
            endpoint: "0.0.0.0:4318"

    processors:
      batch:
        timeout: 1s
        send_batch_size: 1024

    exporters:
      prometheus:
        endpoint: "0.0.0.0:8889"
        const_labels:
          label1: value1
      debug:
        verbosity: detailed

    service:
      pipelines:
        metrics:
          receivers: [otlp]
          processors: [batch]
          exporters: [prometheus, debug]


    and prometheus.yml:
    global:
      scrape_interval: 15s

    scrape_configs:
      - job_name: 'otel-collector'
        static_configs:
          - targets: ['otel-collector:8889']
        metrics_path: '/metrics'

    AND CHANGE THE CONFIG SO THAT OTLP IS USED INSTEAD OF CONSOLE

    """
    set_telemetry_metrics(True)

    for _ in range(15):
        increment_counter("test_operation", smash_config="the best config")

    for _ in range(5):
        increment_counter("test_operation", smash_config="the worse config", success=False)

    @track_usage
    def test_decorated_function():
        pass

    for _ in range(12):
        test_decorated_function()

    # Force flush and give collector time to process
    reader.force_flush()
    time.sleep(1)  # Wait for collector to process


@pytest.mark.integration
def test_otlp_export_errors_not_shown_at_critical_level():
    """Test that errors are not shown in logs when log level is CRITICAL."""
    # Use a StringIO to capture log output directly
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

    # Get all OpenTelemetry loggers
    otel_loggers = [
        logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("opentelemetry")
    ]

    # Store original levels, handlers, and propagate settings
    original_states = []
    for logger in otel_loggers:
        original_states.append((logger, logger.level, logger.handlers.copy(), logger.propagate))
        logger.addHandler(handler)
        # Prevent logs from propagating to console during test
        logger.propagate = False

    try:
        # Set log level to CRITICAL
        set_opentelemetry_log_level("CRITICAL")

        # Generate a metric that will fail to export
        export_metric_to_unreachable_endpoint()

        # Get the log output
        log_output = log_capture.getvalue()

        # Debug output to console (will show in pytest output with -v flag)
        print("\n----- CRITICAL LEVEL LOGS -----")
        print(log_output)
        print("-------------------------------")

        # At CRITICAL level, there should be no ERROR or WARNING messages
        assert "ERROR:" not in log_output, "ERROR logs should not be visible at CRITICAL level"

    finally:
        # Restore original logger states
        for logger, level, handlers, propagate in original_states:
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate


@pytest.mark.integration
def test_otlp_export_errors_shown_at_info_level():
    """Test that errors appear in logs when log level is INFO."""
    # Use a StringIO to capture log output directly
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

    # Get all OpenTelemetry loggers
    otel_loggers = [
        logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("opentelemetry")
    ]

    # Store original levels, handlers, and propagate settings
    original_states = []
    for logger in otel_loggers:
        original_states.append((logger, logger.level, logger.handlers.copy(), logger.propagate))
        logger.addHandler(handler)
        # Prevent logs from propagating to console during test
        logger.propagate = False

    try:
        # Set log level to INFO
        set_opentelemetry_log_level("INFO")

        # Generate a metric that will fail to export
        export_metric_to_unreachable_endpoint()

        # Get the log output
        log_output = log_capture.getvalue()

        # Debug output to console (will show in pytest output with -v flag)
        print("\n----- INFO LEVEL LOGS -----")
        print(log_output)
        print("---------------------------")

        # At INFO level, error messages should be visible
        assert "ERROR:" in log_output, "ERROR logs should be visible at INFO level"

    finally:
        # Restore original logger states
        for logger, level, handlers, propagate in original_states:
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate
