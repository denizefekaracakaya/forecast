"""Email sending via Resend with Jinja2 templates."""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from forecast.config import settings

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "email"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))

_RESEND_AVAILABLE = False
if settings.RESEND_API_KEY:
    try:
        import resend
        resend.api_key = settings.RESEND_API_KEY
        _RESEND_AVAILABLE = True
    except Exception as exc:
        logger.warning("Resend import failed: %s — email disabled", exc)


def _render(template_name: str, **kwargs) -> str:
    template = _env.get_template(template_name)
    return template.render(**kwargs)


def send_verification_email(
    email: str,
    username: str,
    token: str,
    lang: str = "tr",
) -> bool:
    if not _RESEND_AVAILABLE:
        logger.warning("Cannot send verification email — Resend not configured")
        return False
    link = f"{settings.APP_BASE_URL}?verify={token}"
    html = _render("verify.html", username=username, link=link, lang=lang)
    try:
        import resend
        resend.Emails.send({
            "from": f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM}>",
            "to": email,
            "subject": _subject("verify", lang),
            "html": html,
        })
        logger.info("Verification email sent to %s", email)
        return True
    except Exception as exc:
        logger.error("Failed to send verification email: %s", exc)
        return False


def send_password_reset_email(
    email: str,
    username: str,
    token: str,
    lang: str = "tr",
) -> bool:
    if not _RESEND_AVAILABLE:
        logger.warning("Cannot send reset email — Resend not configured")
        return False
    link = f"{settings.APP_BASE_URL}?reset={token}"
    html = _render("reset.html", username=username, link=link, lang=lang)
    try:
        import resend
        resend.Emails.send({
            "from": f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM}>",
            "to": email,
            "subject": _subject("reset", lang),
            "html": html,
        })
        logger.info("Password reset email sent to %s", email)
        return True
    except Exception as exc:
        logger.error("Failed to send reset email: %s", exc)
        return False


def _subject(kind: str, lang: str) -> str:
    if kind == "verify":
        return "E-Forecast — E-posta Doğrulama / Email Verification"
    return "E-Forecast — Şifre Sıfırlama / Password Reset"
