# ===== /urls.py =====
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'GH_ocr'

urlpatterns = [
    path('', views.upload_pdf, name='upload'),
    path('format/', views.format_info, name='format_info'),
    path('download/<str:filename>/', views.download_file, name='download'),
    path('download-template/<str:file_type>/', views.download_template, name='download_template'),
    # 進捗取得API
    path('donation-info/', views.get_donation_info, name='donation_info'),
    path('progress/<str:session_id>/', views.get_progress, name='get_progress'),
]


# urlpatterns の最後に追加
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)