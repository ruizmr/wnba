"""Email helper that will send the rendered daily report via SMTP.

Until SMTP credentials are provisioned this module only validates that the
required environment variables are present. The actual send call is skipped
when running in CI so unit tests do not depend on real network access.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

REQUIRED_ENV_VARS = [
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "SMTP_FROM_ADDR",
]


def _check_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required SMTP env vars: {', '.join(missing)}. "
            "Set them or run in --dry-run mode."
        )


def send_email(
    html_body: str,
    subject: str,
    to_addrs: List[str],
    dry_run: bool = False,
) -> None:
    """Send *html_body* as an email via SMTP.

    When *dry_run* is True the function performs all validations but skips the
    actual network call. This is what the CLI will use during integration
    tests and snapshots.
    """
    _check_env()

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    from_addr = os.getenv("SMTP_FROM_ADDR")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html_body, "html"))

    if dry_run:  # pragma: no cover
        return

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(username, password)
        server.sendmail(from_addr, to_addrs, msg.as_string())