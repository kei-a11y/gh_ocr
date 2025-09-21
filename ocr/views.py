# ===== GH_ocr/views.py =====
import os
import tempfile
import json
import threading
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, Http404
from django.conf import settings
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.http import require_http_methods
from pathlib import Path
import mimetypes
import zipfile
from datetime import datetime
import fitz  # PyMuPDF

from .forms import PDFUploadForm
from .models import PayPayDonation
from .module.WEB_predict_from_pdf_GH import process_pdf_to_excel_with_progress

# 進捗状況を保存するグローバル辞書
progress_storage = {}

def upload_pdf(request):
    """PDFアップロード＆変換ページ（統合版）"""
    
    # URLパラメータから完了状態をチェック
    completed = request.GET.get('completed') == 'true'
    result_file = request.GET.get('file')
    
    if completed:
        # 寄付設定を取得
        donation = PayPayDonation.get_active_donation()
        
        # 完了画面を表示
        return render(request, 'GH_ocr/upload.html', {
            'form': PDFUploadForm(),
            'conversion_complete': True,
            'excel_filename': result_file,
            'donation': donation
        })
    
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data['pdf_file']
            
            try:
                # ファイルを一時保存
                file_path = default_storage.save(
                    f'uploads/{pdf_file.name}', 
                    ContentFile(pdf_file.read())
                )
                
                # PDFのページ数を取得
                full_pdf_path = os.path.join(settings.MEDIA_ROOT, file_path)
                doc = fitz.open(full_pdf_path)
                total_pages = doc.page_count
                doc.close()
                
                # セッションIDを生成（進捗追跡用）
                session_id = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                # 進捗初期化
                progress_storage[session_id] = {
                    'current': 0,
                    'total': total_pages,
                    'status': 'processing',
                    'message': '処理を開始しています...'
                }
                
                # 変換処理を別スレッドで開始
                def process_in_background():
                    try:
                        output_dir = os.path.join(settings.MEDIA_ROOT, "downloads")
                        
                        # 進捗付きでOCR変換実行
                        excel_files = process_pdf_to_excel_with_progress(
                            full_pdf_path, 
                            output_dir=output_dir,
                            progress_callback=lambda current, total, msg: update_progress(session_id, current, total, msg)
                        )
                        
                        if excel_files:
                            # ZIP化
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                            zip_filename = f"{timestamp}_ocr_results.zip"
                            zip_path = os.path.join(output_dir, zip_filename)
                            
                            with zipfile.ZipFile(zip_path, "w") as zipf:
                                for file in excel_files:
                                    zipf.write(file, arcname=os.path.basename(file))
                            
                            print(f"DEBUG: ZIP作成完了 - {zip_filename}")  # デバッグログ
                            
                            # 進捗完了更新
                            progress_storage[session_id] = {
                                'current': total_pages,
                                'total': total_pages,
                                'status': 'completed',
                                'message': '変換が完了しました',
                                'result_file': zip_filename
                            }
                            print(f"DEBUG: 進捗更新完了 - {progress_storage[session_id]}")  # デバッグログ
                        else:
                            print("DEBUG: excel_filesが空です")  # デバッグログ
                            progress_storage[session_id] = {
                                'current': 0,
                                'total': total_pages,
                                'status': 'error',
                                'message': 'PDF変換に失敗しました'
                            }
                            
                        # 一時ファイルを削除
                        if os.path.exists(full_pdf_path):
                            os.remove(full_pdf_path)
                            
                    except Exception as e:
                        print(f"DEBUG: バックグラウンド処理エラー - {str(e)}")  # デバッグログ
                        progress_storage[session_id] = {
                            'current': 0,
                            'total': total_pages,
                            'status': 'error',
                            'message': f'変換エラーが発生しました: {str(e)}'
                        }
                
                # バックグラウンド処理開始
                thread = threading.Thread(target=process_in_background)
                thread.daemon = True
                thread.start()
                
                # 進捗表示画面を返す
                return render(request, 'GH_ocr/upload.html', {
                    'form': PDFUploadForm(),
                    'processing': True,
                    'session_id': session_id,
                    'total_pages': total_pages
                })
                
            except Exception as e:
                messages.error(request, f'変換エラーが発生しました: {str(e)}')
                
            # エラーの場合は一時ファイルを削除
            try:
                full_pdf_path = os.path.join(settings.MEDIA_ROOT, file_path)
                if os.path.exists(full_pdf_path):
                    os.remove(full_pdf_path)
            except:
                pass
    else:
        form = PDFUploadForm()
    
    return render(request, 'GH_ocr/upload.html', {'form': form})


@require_http_methods(["GET"])
def get_donation_info(request):
    """寄付情報を取得するAPI"""
    donation = PayPayDonation.get_active_donation()
    
    if donation and not donation.is_expired():
        # 画像URLを生成
        if donation.paypay_image and donation.paypay_image.name:
            try:
                image_url = request.build_absolute_uri(donation.paypay_image.url)
            except (ValueError, AttributeError):
                image_url = None
        else:
            image_url = None
            
        return JsonResponse({
            'has_donation': True,
            'paypay_image_url': image_url,
            'message': donation.message,
            'expires_at': donation.expires_at.isoformat()
        })
    else:
        return JsonResponse({
            'has_donation': False
        })


def update_progress(session_id, current, total, message):
    """進捗更新用ヘルパー関数"""
    if session_id in progress_storage:
        # 整数に変換して小数点を排除
        current_int = int(round(current)) if isinstance(current, (int, float)) else current
        total_int = int(round(total)) if isinstance(total, (int, float)) else total
        
        progress_storage[session_id].update({
            'current': current_int,
            'total': total_int,
            'message': message
        })
        print(f"DEBUG: 進捗更新 - {session_id}: {current_int}/{total_int} - {message}")


@require_http_methods(["GET"])
def get_progress(request, session_id):
    """進捗状況をJSONで返すAPI"""
    if session_id in progress_storage:
        data = progress_storage[session_id]
        print(f"DEBUG: 進捗取得 - {session_id}: {data}")  # デバッグログ
        return JsonResponse(data)
    else:
        return JsonResponse({
            'current': 0,
            'total': 1,
            'status': 'not_found',
            'message': 'セッションが見つかりません'
        })


def format_info(request):
    """様式情報ページ"""
    return render(request, 'GH_ocr/format_info.html')

def download_file(request, filename):
    """変換結果ファイルのダウンロード"""
    file_path = os.path.join(settings.MEDIA_ROOT, 'downloads', filename)
    if not os.path.exists(file_path):
        raise Http404("ファイルが見つかりません")

    with open(file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type="application/zip")
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

def download_template(request, file_type):
    """テンプレートファイルのダウンロード"""
    file_mapping = {
        'pdf': ('template.pdf', 'application/pdf'),
        'excel': ('template.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        'notes': ('notes.pdf', 'application/pdf'),
    }
    
    if file_type not in file_mapping:
        raise Http404("ファイルタイプが不正です")
    
    filename, content_type = file_mapping[file_type]
    file_path = os.path.join(settings.BASE_DIR, 'static_files', 'documents', filename)
    
    if not os.path.exists(file_path):
        raise Http404("ファイルが見つかりません")
    
    with open(file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type=content_type)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response