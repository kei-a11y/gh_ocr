# ===== /forms.py =====
from django import forms
from django.core.validators import FileExtensionValidator

class PDFUploadForm(forms.Form):
    pdf_file = forms.FileField(
        label='PDFファイル',
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])],
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf'
        })
    )
    
    def clean_pdf_file(self):
        pdf_file = self.cleaned_data.get('pdf_file')
        if pdf_file:
            if pdf_file.size > 10 * 1024 * 1024:  # 10MB
                raise forms.ValidationError('ファイルサイズは10MB以下にしてください。')
        return pdf_file