import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        self.saveState()
        
        # Header
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0284c7"))
        self.drawString(54, 750, "AngioPy Segmentation Application")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#777777"))
        self.drawRightString(558, 750, "Official User & Technical Manual")
        
        self.setStrokeColor(colors.HexColor("#0284c7"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        # Footer
        self.line(54, 50, 558, 50)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#777777"))
        self.drawString(54, 38, "CONFIDENTIAL - CLINICAL & RESEARCH USE ONLY")
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_text)
        
        self.restoreState()

def generate_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    story = []
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("AngioPy Segmentation Manual", title_style))
    story.append(Paragraph("Consolidated Operations, Diagnostics, Data Export, and QCA Analysis Instructions", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Welcome to the official manual for <b>AngioPy Segmentation</b>, a quantitative coronary angiography (QCA) platform. This guide explains all key modules, workflows, and procedures.", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Analyst Authentication & Session Bootstrapping", h1_style))
    story.append(Paragraph("The platform is secured by Firebase Authentication. Analysts must log in to view assignments and synchronize records. When first accessing the app, administrators can configure default analysts via the Admin Panel.", body_style))
    story.append(Paragraph("• <b>Default Credentials:</b> If offline or first initialization, the system uses safe hashed authentication checks.", bullet_style))
    story.append(Paragraph("• <b>User Security:</b> Passwords are encrypted server-side using SHA-256 with specific analyst salts.", bullet_style))
    story.append(Paragraph("• <b>Session Persistence:</b> Session tokens are persisted browser-side until an explicit logout is triggered.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. DICOM Folder Browser & Cache Importer", h1_style))
    story.append(Paragraph("To optimize load speeds, DICOM cases are cached on the VPS disk. The <b>Local Cache Importer</b> sidebar panel offers multiple ways to buffer patient files:", body_style))
    story.append(Paragraph("• <b>Tailscale Shared Folder:</b> Navigate the remote Tailscale mount directly. Status badges indicate progress: To Do, In Progress, and Complete. Click <i>Cache Current Folder</i> to copy to VPS local cache in a background process.", bullet_style))
    story.append(Paragraph("• <b>Mass Prefetch Cases:</b> Multiselect assigned patient cases on Tailscale. The system estimates size, checks against disk limits, and runs copies sequentially in a background thread.", bullet_style))
    story.append(Paragraph("• <b>Upload Local Files:</b> Upload individual DICOM files or drag whole directories. The folder name is automatically detected from DICOM headers.", bullet_style))
    story.append(Paragraph("• <b>Upload ZIP Archive:</b> Upload ZIP compressed cases. The server extracts files on the fly.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("3. DICOM Series Selection Grid", h1_style))
    story.append(Paragraph("Once a patient folder is loaded into cache, the workspace switches to the Selection Grid mode:", body_style))
    story.append(Paragraph("• <b>Frame Auto-Detection:</b> The app scans frames and automatically identifies the optimal contrast frame containing peak opacification.", bullet_style))
    story.append(Paragraph("• <b>Sequence Configuration:</b> Check <i>Chosen for Analysis</i> to select sequences. Tag the Procedure Phase (PRE-PCI or POST-PCI), Vessel System (LAD, LCx, RCA) and Segment (AHA classification code).", bullet_style))
    story.append(Paragraph("• <b>Animation Control:</b> Click <i>Play</i> to run the angiographic cine-loop. Click <i>Stop</i> to freeze.", bullet_style))
    story.append(Paragraph("• <b>Direct Download:</b> Use <i>Download DICOM</i> to download the raw DICOM file directly from the VPS cache.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("4. QCA Interactive Analysis Workspace", h1_style))
    story.append(Paragraph("Clicking <i>Analyze (QCA)</i> loads the interactive workstation containing algorithmic and manual tools:", body_style))
    story.append(Paragraph("• <b>Centerline and Contour Detection:</b> The system runs automatic edge detection. Manual sliders allow tweaking landmarks: Proximal Reference, Distal Reference, and Minimum Lumen Diameter (MLD) positions.", bullet_style))
    story.append(Paragraph("• <b>Edge Reference Modes:</b> Analysts can choose between Interpolated Reference (centerline arc-length cumLen) or Mean, Max, and Manual reference computation.", bullet_style))
    story.append(Paragraph("• <b>PCI Options:</b> Adjust FFR wire indicators, other distal lesions (>50% distal to FFR/DES), and procedure phase offsets.", bullet_style))
    story.append(Paragraph("• <b>Save Sequence:</b> Generate the sequence report. The PDF is saved locally, and numerical outputs are synchronized to the Firestore <i>analysis_results</i> collection.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("5. Saved Stenoses Checklist & Master PDF Persistence", h1_style))
    story.append(Paragraph("• <b>Saved Checklist:</b> A widget reads all reports from Firestore under the current patient ID, ensuring that all necessary vessel segments have been analyzed prior to completion.", bullet_style))
    story.append(Paragraph("• <b>Persistent Master PDF:</b> Clicking <i>Export Master Patient PDF</i> consolidates all saved individual sequence PDFs on-the-fly, allowing session restoration after refreshing.", bullet_style))
    story.append(Paragraph("• <b>Finish Patient Analysis:</b> Confirms session closure. The consolidated PDF is uploaded to Firebase Storage and the patient ID is marked as Completed.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("6. Administrator Capabilities & Data Export", h1_style))
    story.append(Paragraph("Administrators can navigate to the Admin Panel to oversee operations:", body_style))
    story.append(Paragraph("• <b>Analyst Assignments:</b> Assign cases or entire sites (hospitals/centers) to specific analysts.", bullet_style))
    story.append(Paragraph("• <b>Progress Metrics:</b> Access database statistics, disk file metrics, and individual analyst completion rates.", bullet_style))
    story.append(Paragraph("• <b>Data Export & Consolidation:</b> Export comprehensive analysis records directly to structured Excel spreadsheets (<i>AngioPy_Analysis_Results_2026.xlsx</i>) and download encrypted packages of all clinical PDF reports (<i>AngioPy_All_PDF_Reports_2026.zip</i>).", bullet_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF manual successfully written to {output_path}")

if __name__ == "__main__":
    generate_pdf("/tmp/AngioPy_User_Manual.pdf")
