import sys; sys.stdout.reconfigure(encoding='utf-8') 
from pathlib import Path 
from reportlab.lib import colors 
from reportlab.lib.pagesizes import A4 
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle 
from reportlab.lib.units import cm 
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer 
O=Path(__file__).resolve().parent / "StorePulse_AI_Demo_Script.pdf" 
s=getSampleStyleSheet() 
T=ParagraphStyle("T2",parent=s["Title"],fontSize=24,leading=30,spaceAfter=2,textColor=colors.HexColor("#2563EB"),alignment=1) 
SUB=ParagraphStyle("Sub",parent=s["Normal"],fontSize=12,leading=16,spaceAfter=18,textColor=colors.HexColor("#475569"),alignment=1) 
TM=ParagraphStyle("Tm",parent=s["Normal"],fontSize=8,leading=10,textColor=colors.HexColor("#64748B"),spaceBefore=6,spaceAfter=1) 
B=ParagraphStyle("B",parent=s["Normal"],fontSize=10,leading=14,spaceAfter=4) 
SC=ParagraphStyle("Sc",parent=s["Normal"],fontSize=9,leading=12,leftIndent=12,textColor=colors.HexColor("#1E293B"),backColor=colors.HexColor("#F1F5F9"),borderPadding=6) 
