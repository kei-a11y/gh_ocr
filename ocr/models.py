from django.db import models
from django.utils import timezone
import os

def paypay_image_path(instance, filename):
    """PayPay画像の保存パス"""
    return f'paypay_codes/{filename}'

class PayPayDonation(models.Model):
    """PayPay寄付コード管理"""
    paypay_image = models.ImageField(
        upload_to=paypay_image_path,
        verbose_name="PayPayコード画像",
        help_text="PayPayのQRコード画像をアップロードしてください",
        blank=True,
        null=True
    )
    message = models.TextField(
        verbose_name="寄付メッセージ",
        default="このサービスの維持・改善にご協力ください",
        help_text="寄付をお願いする際に表示するメッセージ"
    )
    is_active = models.BooleanField(
        verbose_name="有効",
        default=True,
        help_text="現在使用中の寄付コードかどうか"
    )
    created_at = models.DateTimeField(
        verbose_name="作成日時",
        auto_now_add=True
    )
    expires_at = models.DateTimeField(
        verbose_name="有効期限",
        help_text="PayPayコードの有効期限"
    )
    
    class Meta:
        verbose_name = "PayPay寄付設定"
        verbose_name_plural = "PayPay寄付設定"
        ordering = ['-created_at']
    
    def __str__(self):
        status = "有効" if self.is_active else "無効"
        return f"PayPay寄付コード ({status}) - {self.created_at.strftime('%Y/%m/%d')}"
    
    def is_expired(self):
        """有効期限切れかどうかを判定"""
        return timezone.now() > self.expires_at
    
    @classmethod
    def get_active_donation(cls):
        """現在有効な寄付設定を取得"""
        try:
            return cls.objects.filter(
                is_active=True,
                expires_at__gt=timezone.now()
            ).first()
        except cls.DoesNotExist:
            return None