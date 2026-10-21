"""Reproducible ReportLab export of the manuscript's restricted Markdown.

Run using the already bundled document Python runtime. Supports paragraphs,
headings, simple lists, links, images, code blocks and Markdown tables. Equations
should be readable Unicode or standalone scientific figure assets. The PDF is
an editable-source manuscript export, not a journal-specific template.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Image,Table,
                               TableStyle,Preformatted,KeepTogether)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inline(value):
    value=html.escape(value.replace('\u2011','-').replace('—','-').replace('–','-'))
    value=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',value)
    value=re.sub(r'`([^`]+)`',r'<font name="DejaVuSansMono">\1</font>',value)
    value=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#166274">\1</link>',value)
    value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1',value)
    return value


def build(source,destination):
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    for name,file in [('DejaVuSerif','DejaVuSerif.ttf'),('DejaVuSerif-Bold','DejaVuSerif-Bold.ttf'),
        ('DejaVuSans','DejaVuSans.ttf'),('DejaVuSans-Bold','DejaVuSans-Bold.ttf'),('DejaVuSansMono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('DejaVuSerif',normal='DejaVuSerif',bold='DejaVuSerif-Bold')
    pdfmetrics.registerFontFamily('DejaVuSans',normal='DejaVuSans',bold='DejaVuSans-Bold')
    styles={
        'body':ParagraphStyle('body',fontName='DejaVuSerif',fontSize=10,leading=14,spaceAfter=7,
            allowWidows=0,allowOrphans=0),
        'h1':ParagraphStyle('h1',fontName='DejaVuSans-Bold',fontSize=18,leading=22,spaceAfter=14),
        'h2':ParagraphStyle('h2',fontName='DejaVuSans-Bold',fontSize=12,leading=16,spaceBefore=12,spaceAfter=7,keepWithNext=True),
        'h3':ParagraphStyle('h3',fontName='DejaVuSans-Bold',fontSize=10.5,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=True),
        'caption':ParagraphStyle('caption',fontName='DejaVuSans',fontSize=8.5,leading=11,spaceAfter=10),
        'cell':ParagraphStyle('cell',fontName='DejaVuSans',fontSize=8,leading=10),
        'code':ParagraphStyle('code',fontName='DejaVuSansMono',fontSize=7,leading=9,spaceAfter=8),
    }
    width=A4[0]-100
    lines=source.read_text().splitlines()
    title=next((line[2:].strip() for line in lines if line.startswith('# ')), 'Simulation manuscript')
    doc=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=47,bottomMargin=48,
        title=title,author='Anonymous manuscript',
        subject='Simulation study; scientific scope and limitations stated in text')
    story=[];assets={str(source):sha(source)};i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.extend(textwrap.wrap(lines[i],width=94,replace_whitespace=False,drop_whitespace=False) or [''])
                i+=1
            story.append(Preformatted('\n'.join(code),styles['code']));i+=1;continue
        image_match=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if image_match:
            caption,path=image_match.groups();asset=(source.parent/path).resolve();assets[str(asset)]=sha(asset)
            height_limit=580 if asset.name=='experimental_design.png' else (360 if asset.name=='work_and_forecast_loss.png' else 320)
            item=Image(str(asset));ratio=min(width/item.imageWidth,height_limit/item.imageHeight)
            item.drawWidth=item.imageWidth*ratio;item.drawHeight=item.imageHeight*ratio
            block=[item,Spacer(1,5),Paragraph(inline(caption),styles['caption'])]
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name in ('h2','h3'):
                block.insert(0,story.pop())
            story.append(KeepTogether(block));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                i+=1
            if any(len(row)!=len(rows[0]) for row in rows):raise ValueError('Nonrectangular Markdown table')
            body=[[Paragraph(inline(c),styles['cell']) for c in row] for row in rows]
            table=Table(body,colWidths=[width/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4ecef')),
                ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#71888f')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d0dadd'))]))
            story.extend([table,Spacer(1,9)]);continue
        heading=re.match(r'^(#{1,3})\s+(.+)',line)
        if heading:
            story.append(Paragraph(inline(heading.group(2)),styles[f'h{len(heading.group(1))}']));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|```|[-*] |\d+\. )',lines[i].strip()):
            paragraph.append(lines[i].strip());i+=1
        text=' '.join(paragraph)
        if text.startswith('- '):text='• '+text[2:]
        story.append(Paragraph(inline(text),styles['body']))
    def page(canvas,document):
        canvas.saveState();canvas.setFont('DejaVuSans',7.5);canvas.setFillColor(colors.HexColor('#52656b'))
        canvas.drawString(50,A4[1]-27,'Material history and compositional prediction | Simulation manuscript')
        canvas.drawString(50,27,'Reproducible local study - scope and numerical limitations stated in the manuscript')
        canvas.drawRightString(A4[0]-50,27,str(document.page));canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)
    assets[str(Path(__file__).resolve())]=sha(__file__)
    destination.with_suffix('.provenance.json').write_text(json.dumps(dict(sources_sha256=assets,
        pdf_sha256=sha(destination),renderer='ReportLab',template='A4 single-column scientific manuscript'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists() or args.out.with_suffix('.provenance.json').exists():
        raise ValueError('Fresh PDF and provenance destination required')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    build(args.source.resolve(),args.out.resolve());print(args.out)


if __name__=='__main__':main()
