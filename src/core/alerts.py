"""
Real-Time Alerts Module

Send notifications when high-conviction signals are detected.
Supports Slack, Discord, Email, and Telegram.
"""

from __future__ import annotations
import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass
import requests

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alerts."""
    enabled: bool = False
    channels: list[str] = None  # ["email", "telegram", "desktop", "file", "slack", "discord"]
    
    # Slack
    slack_webhook: Optional[str] = None
    
    # Discord
    discord_webhook: Optional[str] = None
    
    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: list[str] = None
    
    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Desktop notifications (macOS/Windows/Linux)
    desktop_sound: bool = True
    
    # File-based alerts (local log)
    alert_log_path: str = "outputs/alerts.log"
    
    # Triggers
    trigger_all_three_overlap: bool = True
    trigger_weekly_pro30_overlap: bool = True
    trigger_high_composite_score: float = 7.0
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = []
        if self.to_addresses is None:
            self.to_addresses = []
        
        # Load from environment variables
        self.slack_webhook = self.slack_webhook or os.environ.get("SLACK_WEBHOOK_URL")
        self.discord_webhook = self.discord_webhook or os.environ.get("DISCORD_WEBHOOK_URL")
        self.smtp_user = self.smtp_user or os.environ.get("SMTP_USER")
        self.smtp_password = self.smtp_password or os.environ.get("SMTP_PASSWORD")
        self.telegram_bot_token = self.telegram_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = self.telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")


class AlertManager:
    """Manages sending alerts across multiple channels."""
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._handlers: dict[str, Callable] = {
            "slack": self._send_slack,
            "discord": self._send_discord,
            "email": self._send_email,
            "telegram": self._send_telegram,
            "desktop": self._send_desktop,
            "file": self._send_file,
        }
    
    def send_alert(
        self,
        title: str,
        message: str,
        data: Optional[dict] = None,
        channels: Optional[list[str]] = None,
        priority: str = "normal"
    ) -> dict[str, bool]:
        """
        Send alert to configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            data: Additional structured data
            channels: Override default channels
            priority: "low", "normal", "high"
        
        Returns:
            Dict mapping channel -> success status
        """
        if not self.config.enabled:
            logger.debug("Alerts disabled, skipping")
            return {}
        
        channels = channels or self.config.channels
        results = {}
        
        for channel in channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    success = handler(title, message, data, priority)
                    results[channel] = success
                except Exception as e:
                    logger.error(f"Failed to send {channel} alert: {e}")
                    results[channel] = False
            else:
                logger.warning(f"Unknown alert channel: {channel}")
                results[channel] = False
        
        return results
    
    def _send_slack(
        self,
        title: str,
        message: str,
        data: Optional[dict],
        priority: str
    ) -> bool:
        """Send alert to Slack webhook."""
        webhook_url = self.config.slack_webhook
        if not webhook_url:
            logger.warning("Slack webhook not configured")
            return False
        
        # Build Slack blocks
        emoji = {"low": "📊", "normal": "📈", "high": "🚨"}.get(priority, "📈")
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {title}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message}
            }
        ]
        
        if data:
            # Add data fields
            fields = []
            for key, value in list(data.items())[:10]:  # Limit fields
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}"
                })
            
            if fields:
                blocks.append({
                    "type": "section",
                    "fields": fields[:10]  # Slack limit
                })
        
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"Sent at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            }]
        })
        
        payload = {"blocks": blocks}
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        return response.status_code == 200
    
    def _send_discord(
        self,
        title: str,
        message: str,
        data: Optional[dict],
        priority: str
    ) -> bool:
        """Send alert to Discord webhook."""
        webhook_url = self.config.discord_webhook
        if not webhook_url:
            logger.warning("Discord webhook not configured")
            return False
        
        # Build Discord embed
        color = {"low": 0x808080, "normal": 0x00FF00, "high": 0xFF0000}.get(priority, 0x00FF00)
        
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Momentum Scanner"}
        }
        
        if data:
            embed["fields"] = [
                {"name": k, "value": str(v)[:1024], "inline": True}
                for k, v in list(data.items())[:25]  # Discord limit
            ]
        
        payload = {"embeds": [embed]}
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        return response.status_code == 204
    
    def _send_email(
        self,
        title: str,
        message: str,
        data: Optional[dict],
        priority: str
    ) -> bool:
        """Send alert via email."""
        if not all([
            self.config.smtp_user,
            self.config.smtp_password,
            self.config.from_address,
            self.config.to_addresses
        ]):
            logger.warning("Email not fully configured")
            return False
        
        # Build email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{priority.upper()}] {title}"
        msg["From"] = self.config.from_address
        msg["To"] = ", ".join(self.config.to_addresses)
        
        # Plain text version
        text_content = f"{title}\n\n{message}"
        if data:
            text_content += "\n\nDetails:\n"
            for k, v in data.items():
                text_content += f"  {k}: {v}\n"
        
        # HTML version
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: {'#FF0000' if priority == 'high' else '#333'};">{title}</h2>
            <p style="font-size: 14px; line-height: 1.6;">{message.replace(chr(10), '<br>')}</p>
        """
        
        if data:
            html_content += """
            <table style="border-collapse: collapse; margin-top: 20px;">
                <tr><th style="text-align: left; padding: 8px; background: #f0f0f0;">Field</th>
                    <th style="text-align: left; padding: 8px; background: #f0f0f0;">Value</th></tr>
            """
            for k, v in data.items():
                html_content += f"""
                <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">{k}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{v}</td></tr>
                """
            html_content += "</table>"
        
        html_content += f"""
            <p style="font-size: 12px; color: #888; margin-top: 30px;">
                Sent at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
            </p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))
        
        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False
    
    def _send_telegram(
        self,
        title: str,
        message: str,
        data: Optional[dict],
        priority: str
    ) -> bool:
        """Send alert to Telegram."""
        token = self.config.telegram_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = self.config.telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")

        logger.debug(
            "Telegram config: token_present=%s chat_id=%s",
            bool(token),
            chat_id if chat_id else None,
        )

        missing = []
        if not token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        if missing:
            logger.warning(f"Telegram not configured (missing {', '.join(missing)})")
            return False
        
        emoji = {"low": "📊", "normal": "📈", "high": "🚨"}.get(priority, "📈")
        
        # Escape Markdown special characters to prevent parse errors
        def escape_md(s: str) -> str:
            """Escape Markdown special characters for Telegram."""
            if not s:
                return ""
            # Characters that need escaping in Telegram Markdown
            for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                s = str(s).replace(char, f'\\{char}')
            return s
        
        # Build message - use plain text to avoid Markdown issues
        text = f"{emoji} {title}\n\n{message}"
        
        if data:
            text += "\n\nDetails:"
            for k, v in data.items():
                text += f"\n• {k}: {v}"
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        # Try without Markdown first (more reliable)
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Telegram alert sent successfully: {title}")
                return True
            else:
                # Log the actual error from Telegram
                try:
                    error_data = response.json()
                    error_desc = error_data.get('description', 'Unknown error')
                    logger.error(f"Telegram API error ({response.status_code}): {error_desc}")
                except Exception:
                    logger.error(f"Telegram API error ({response.status_code}): {response.text[:200]}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("Telegram request timed out")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram request failed: {e}")
            return False
    
    def _send_desktop(
        self,
        title: str,
        message: str,
        data: Optional[dict],
        priority: str
    ) -> bool:
        """Send desktop notification (macOS/Windows/Linux)."""
        import platform
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                # Use osascript for native macOS notifications
                import subprocess
                
                # Escape quotes in message
                safe_title = title.replace('"', '\\"')
                safe_message = message.replace('"', '\\"').replace('\n', ' ')
                
                script = f'display notification "{safe_message}" with title "{safe_title}"'
                if self.config.desktop_sound:
                    script += ' sound name "Glass"'
                
                subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    timeout=5
                )
                return True
                
            elif system == "Windows":
                # Try Windows toast notifications
                try:
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(title, message, duration=10, threaded=True)
                    return True
                except ImportError:
                    # Fallback to basic Windows notification
                    import ctypes
                    ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)
                    return True
                    
            elif system == "Linux":
                # Use notify-send on Linux
                import subprocess
                subprocess.run(
                    ["notify-send", title, message],
                    capture_output=True,
                    timeout=5
                )
                return True
                
        except Exception as e:
            logger.error(f"Desktop notification failed: {e}")
            return False
        
        return False
    
    def _send_file(
        self,
        title: str,
        message: str,
        data: Optional[dict],
        priority: str
    ) -> bool:
        """Write alert to local log file."""
        try:
            from pathlib import Path
            
            log_path = Path(self.config.alert_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"[{timestamp}] [{priority.upper()}] {title}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{message}\n")
                
                if data:
                    f.write("\nDetails:\n")
                    for k, v in data.items():
                        f.write(f"  {k}: {v}\n")
                
                f.write("\n")
            
            return True
            
        except Exception as e:
            logger.error(f"File alert failed: {e}")
            return False


# Convenience functions

def send_overlap_alert(
    overlap_type: str,
    tickers: list[str],
    date_str: str,
    config: Optional[AlertConfig] = None
) -> dict[str, bool]:
    """Send alert for overlap detection."""
    manager = AlertManager(config)
    
    if overlap_type == "all_three" and not manager.config.trigger_all_three_overlap:
        return {}
    if overlap_type == "weekly_pro30" and not manager.config.trigger_weekly_pro30_overlap:
        return {}
    
    emoji_map = {
        "all_three": "⭐",
        "weekly_pro30": "🔥",
        "weekly_movers": "📈",
        "pro30_movers": "💎"
    }
    
    title_map = {
        "all_three": "ALL THREE Overlap Detected!",
        "weekly_pro30": "Weekly + Pro30 Overlap",
        "weekly_movers": "Weekly + Movers Overlap",
        "pro30_movers": "Pro30 + Movers Overlap"
    }
    
    emoji = emoji_map.get(overlap_type, "📊")
    title = title_map.get(overlap_type, f"{overlap_type} Overlap")
    
    message = f"{emoji} {len(tickers)} ticker(s) found in {overlap_type.replace('_', ' ')} overlap:\n"
    message += ", ".join(tickers[:10])
    if len(tickers) > 10:
        message += f" (+{len(tickers) - 10} more)"
    
    priority = "high" if overlap_type == "all_three" else "normal"
    
    return manager.send_alert(
        title=title,
        message=message,
        data={
            "Date": date_str,
            "Overlap Type": overlap_type,
            "Ticker Count": len(tickers),
            "Tickers": ", ".join(tickers)
        },
        priority=priority
    )


def send_high_score_alert(
    ticker: str,
    score: float,
    rank: int,
    date_str: str,
    config: Optional[AlertConfig] = None
) -> dict[str, bool]:
    """Send alert for high composite score."""
    manager = AlertManager(config)
    
    if score < manager.config.trigger_high_composite_score:
        return {}
    
    return manager.send_alert(
        title=f"High Score Alert: {ticker}",
        message=f"🏆 {ticker} ranked #{rank} with composite score {score:.2f}",
        data={
            "Date": date_str,
            "Ticker": ticker,
            "Rank": rank,
            "Composite Score": f"{score:.2f}"
        },
        priority="high" if score >= 8.0 else "normal"
    )


def send_run_summary_alert(
    date_str: str,
    weekly_count: int,
    pro30_count: int,
    movers_count: int,
    overlaps: dict,
    config: Optional[AlertConfig] = None,
    weekly_tickers: Optional[list] = None,
    pro30_tickers: Optional[list] = None,
    movers_tickers: Optional[list] = None,
    model_health: Optional[dict] = None,
    weekly_top5_data: Optional[list] = None,
    hybrid_top3: Optional[list] = None,
    primary_label: str = "Weekly",
    primary_candidates_count: Optional[int] = None,
    position_alerts: Optional[dict] = None,
    regime: Optional[str] = None,
) -> dict[str, bool]:
    """Send comprehensive daily scan summary - ONE message with all details."""
    manager = AlertManager(config)
    
    all_three = overlaps.get("all_three", [])
    primary_pro30 = overlaps.get("primary_pro30", overlaps.get("weekly_pro30", []))
    primary_movers = overlaps.get("primary_movers", overlaps.get("weekly_movers", []))
    pro30_movers = overlaps.get("pro30_movers", [])
    
    # Ensure lists
    weekly_tickers = list(weekly_tickers) if weekly_tickers else []
    pro30_tickers = list(pro30_tickers) if pro30_tickers else []
    movers_tickers = list(movers_tickers) if movers_tickers else []
    weekly_top5_data = weekly_top5_data or []
    hybrid_top3 = hybrid_top3 or []
    
    # Build comprehensive message similar to CLI output
    lines = [
        f"{'='*35}",
        f"SCAN COMPLETE - {date_str}",
        f"{'='*35}",
    ]
    if regime:
        lines.append(f"Regime: {regime}")
    lines.append(f"Primary Strategy: {primary_label}")
    if primary_candidates_count is not None:
        lines.append(f"Primary Candidates: {primary_candidates_count}")
    lines.append("")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HYBRID TOP 3 (Most Important - Best across all models by weighted hit rate)
    # ═══════════════════════════════════════════════════════════════════════════════
    if hybrid_top3:
        lines.append("🏆 HYBRID TOP 3 (Best Picks):")
        for item in hybrid_top3[:3]:
            ticker = item.get("ticker", "?")
            hybrid_score = item.get("hybrid_score", 0)
            sources = ", ".join(item.get("sources", []))
            confidence = item.get("confidence", "?")
            name = item.get("name", "")
            if name:
                name = name[:20]  # Truncate for Telegram
            
            lines.append(f"  {item.get('rank', '?')}. {ticker} ({sources})")
            lines.append(f"     Score: {hybrid_score:.1f} pts | {confidence}")
        lines.append("")

    # Primary Top 5
    if weekly_top5_data:
        lines.append(f"⭐ {primary_label} TOP PICKS:")
        for item in weekly_top5_data[:5]:
            ticker = item.get("ticker", "?")
            score = item.get("composite_score")
            if score is None:
                score = item.get("swing_score", 0) or 0
            try:
                score = float(score)
            except Exception:
                score = 0.0
            verdict = item.get("verdict") or item.get("confidence", "")
            lines.append(f"  - {ticker}: {score:.2f} {verdict}".rstrip())
        lines.append("")
    
    # Overlaps Section
    if all_three or primary_pro30 or primary_movers or pro30_movers:
        lines.append("🎯 OVERLAPS (High Conviction):")
        if all_three:
            lines.append(f"  ⭐ ALL THREE: {', '.join(all_three)}")
        if primary_pro30:
            non_triple = [t for t in primary_pro30 if t not in all_three]
            if non_triple:
                lines.append(f"  🔥 {primary_label}+Pro30: {', '.join(non_triple)}")
        if primary_movers:
            non_other = [t for t in primary_movers if t not in all_three]
            if non_other:
                lines.append(f"  📈 {primary_label}+Movers: {', '.join(non_other)}")
        if pro30_movers:
            non_other = [t for t in pro30_movers if t not in all_three]
            if non_other:
                lines.append(f"  💎 Pro30+Movers: {', '.join(non_other)}")
        lines.append("")
    
    # Summary counts
    lines.append("📊 SUMMARY:")
    lines.append(f"  {primary_label}: {weekly_count} | Pro30: {pro30_count} | Movers: {movers_count}")
    lines.append("")
    
    # Model Health
    if model_health:
        health_status = model_health.get("status", "Unknown")
        lines.append(f"📈 MODEL: {health_status}")
        
        hit_rate = model_health.get("hit_rate")
        win_rate = model_health.get("win_rate")
        if hit_rate is not None and win_rate is not None:
            lines.append(f"  Hit: {hit_rate * 100:.0f}% | Win: {win_rate * 100:.0f}%")
        
        # Strategy breakdown (compact)
        strategies = model_health.get("strategies", [])
        if strategies:
            strat_parts = []
            for s in strategies[:3]:
                name = s.get("name", "?")
                s_hit = s.get("hit_rate", 0) * 100
                strat_parts.append(f"{name}:{s_hit:.0f}%")
            lines.append(f"  {' | '.join(strat_parts)}")
        lines.append("")

    if position_alerts:
        lines.append("📌 POSITION ALERTS:")
        lines.append(f"  Total: {position_alerts.get('count', 0)} | High: {position_alerts.get('high', 0)} | Warning: {position_alerts.get('warning', 0)}")
        samples = position_alerts.get("sample") or []
        for msg in samples[:3]:
            lines.append(f"  - {msg}")
        lines.append("")
    
    lines.append(f"{'='*35}")
    
    # Determine priority based on hybrid top 3 and overlaps
    if all_three:
        priority = "high"
    elif hybrid_top3 and any("Pro30" in item.get("sources", []) for item in hybrid_top3):
        priority = "high"
    elif primary_pro30:
        priority = "normal"
    else:
        priority = "low"
    
    message = "\n".join(lines)
    
    return manager.send_alert(
        title=f"Scan Complete: {date_str} ({primary_label})",
        message=message,
        data=None,  # All info is in the message
        priority=priority
    )
