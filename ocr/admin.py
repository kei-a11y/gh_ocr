# /admin.py
from django.contrib import admin
from django.utils import timezone
from django.utils.html import format_html
from .models import PayPayDonation

@admin.register(PayPayDonation)
class PayPayDonationAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'message_short', 'is_active', 'expires_at', 'is_expired_status', 'image_preview_small', 'created_at')
    list_filter = ('is_active', 'created_at', 'expires_at')
    search_fields = ('message',)
    ordering = ('-created_at',)
    readonly_fields = ('image_preview',)
    
    fieldsets = (
        ('基本情報', {
            'fields': ('paypay_image', 'image_preview', 'message', 'is_active')
        }),
        ('有効期限', {
            'fields': ('expires_at',)
        }),
    )
    
    def message_short(self, obj):
        """メッセージの短縮版を表示"""
        if len(obj.message) > 30:
            return obj.message[:30] + '...'
        return obj.message
    message_short.short_description = '寄付メッセージ'
    
    def is_expired_status(self, obj):
        """有効期限切れ状態を表示"""
        if obj.is_expired():
            return '期限切れ'
        else:
            return '有効'
    is_expired_status.short_description = '期限状態'
    is_expired_status.admin_order_field = 'expires_at'
    
    def image_preview(self, obj):
        """管理画面で画像プレビューを表示"""
        if obj.paypay_image and obj.paypay_image.name:
            return format_html('<img src="{}" style="max-width: 200px; max-height: 200px;" />', obj.paypay_image.url)
        return '画像なし'
    image_preview.short_description = 'プレビュー'
    
    def image_preview_small(self, obj):
        """リスト表示用の小さなプレビュー"""
        if obj.paypay_image and obj.paypay_image.name:
            return format_html('<img src="{}" style="max-width: 50px; max-height: 50px;" />', obj.paypay_image.url)
        return '画像なし'
    image_preview_small.short_description = '画像'
    
    def save_model(self, request, obj, form, change):
        """新しい寄付設定を有効にする際、他を無効にする"""
        if obj.is_active:
            # 新しく有効にする場合、他の設定を無効にする
            PayPayDonation.objects.filter(is_active=True).update(is_active=False)
        super().save_model(request, obj, form, change)