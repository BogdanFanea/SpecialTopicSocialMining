"""Build the two-page Assignment 2 report using the single-column Assignment 1 layout."""
import json

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

from main import HERE


def main():
    folder = HERE / "artifacts"
    results = json.loads((folder / "results.json").read_text())
    analysis = json.loads((folder / "analysis.json").read_text())
    linear, gated, *op_trials = results["trials"]
    op = next(t for t in op_trials if t["lambda"] == results["selected_op_lambda"])
    image, text = (analysis["baseline_dev"][key] for key in ("clip", "mpnet"))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=10, leading=11.7,
                              alignment=TA_JUSTIFY, spaceAfter=4))
    styles.add(ParagraphStyle("HeaderLine", fontName="Times-Roman", fontSize=10, leading=12,
                              alignment=TA_CENTER, spaceAfter=3))
    styles.add(ParagraphStyle("PaperTitle", fontName="Times-Bold", fontSize=16, leading=19,
                              alignment=TA_CENTER, spaceAfter=5))
    styles.add(ParagraphStyle("Section", fontName="Times-Bold", fontSize=10.5, leading=12.5,
                              spaceBefore=5, spaceAfter=2))
    styles.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=9.5, leading=11))
    styles.add(ParagraphStyle("Caption", fontName="Times-Roman", fontSize=9, leading=11,
                              spaceAfter=4))
    styles.add(ParagraphStyle("ReferenceTitle", fontName="Times-Bold", fontSize=14,
                              leading=17, spaceBefore=5, spaceAfter=2))
    styles.add(ParagraphStyle("Reference", fontName="Times-Roman", fontSize=9, leading=10.4,
                              leftIndent=11, firstLineIndent=-13))
    story = []

    def para(value, style="Body"):
        story.append(Paragraph(value, styles[style]))

    def table(data, widths, highlight=None, alignment="RIGHT"):
        tbl = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT")
        instructions = [
            ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (-1, -1), alignment),
            ("LEFTPADDING", (0, 0), (0, -1), 0),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ]
        if highlight:
            instructions.append(("FONTNAME", (0, highlight), (-1, highlight), "Times-Bold"))
        tbl.setStyle(TableStyle(instructions))
        tbl.spaceAfter = 4
        story.append(tbl)

    repo = "https://github.com/BogdanFanea/SpecialTopicSocialMining"
    para("Special Topics: Social Media Mining and Analysis – Assignment 2", "HeaderLine")
    para("Multimodal Hateful Meme Classification", "PaperTitle")
    para("Bogdan Fanea &nbsp;&nbsp;&nbsp;&nbsp; Lena Bryll &nbsp;&nbsp;&nbsp;&nbsp; Natalie Stadler",
         "HeaderLine")
    para(f'<font name="Courier" size="8"><link href="{repo}">{repo}</link></font>', "HeaderLine")
    para("We adapted Fusion and Orthogonal Projection (FOP) [1, 2] to classify memes using both "
         "the image and its supplied text. We reuse frozen embeddings from Assignment 1 and compare "
         "weighted addition, gated fusion, and gated fusion with an orthogonal projection loss.")
    para("1. Data and evaluation labels", "Section")
    table([["Split", "Examples", "Non-hateful", "Hateful"],
           ["Training", "8,500", "5,450", "3,050"],
           ["Validation (dev)", "500", "250", "250"],
           ["Held-out test", "1,000", "—", "—"]], [86, 22, 34, 30])
    para("We kept the course Hateful Memes splits [3] and their supplied training and validation "
         "labels. The splits have no shared IDs. For each split, we checked the cached feature "
         "dimensions, annotation hashes and exact ID membership, then joined image and text "
         "embeddings by meme ID. The course test file has no labels. We report validation AUROC "
         "and export one held-out probability per ID.")
    para("2. Encoders and preprocessing", "Section")
    table([["Input", "Exact pretrained checkpoint", "Dim.", "Pooling"],
           ["Image", Paragraph('<font name="Courier" size="8.5">openai/clip-vit-base-patch32</font>',
                               styles["Cell"]), "512", "Vision CLS + projection"],
           ["Text", Paragraph('<font name="Courier" size="8.5">sentence-transformers/all-mpnet-base-v2</font>',
                              styles["Cell"]), "768", "Masked mean + L2"]],
          [16, 100, 11, 45], alignment="LEFT")
    para("CLIP and MPNet had the highest mean validation AUROC in their respective modalities in "
         "Assignment 1. Each fusion model uses the same frozen pair. CLIP [4] takes RGB pixels "
         "after bicubic resizing, a 224 × 224 center crop and checkpoint channel normalization. "
         "MPNet [5] uses the supplied transcript, its matching tokenizer, dynamic padding and a "
         "384-token limit. Masked pooling excludes padding, then the embedding is L2-normalized. "
         "Exact revisions are recorded in the checkpoint metadata and Assignment 1 configuration.")
    para("3. Fusion, training and model selection", "Section")
    para("Each projection maps its input to 128 dimensions with Linear, BatchNorm, ReLU, dropout "
         "(p = 0.5), then rowwise L2 normalization. Following FOP [1, 2], the gated model concatenates "
         "the projected vectors u and v. A Linear(256,128), BatchNorm, ReLU and Linear(128,128) "
         "network followed by sigmoid produces gate g. Fusion combines tanh(u) and tanh(v) "
         "elementwise with weights g and 1 minus g. The linear addition baseline uses two learned "
         "scalar weights on u and v. A Linear(128,2) head predicts raw logits. Softmax class 1 is the "
         "hateful probability at inference.")
    para("All variants minimize cross-entropy on integer labels. For CE+OP we add a weighted "
         "penalty on L2-normalized fused vectors. It is one minus the mean same-class cosine "
         "similarity, plus 0.7 times the mean absolute cross-class cosine similarity. Same-class "
         "self-pairs are excluded and empty means contribute zero. The pairs are whole memes, "
         "with each original image still paired to its transcript.")
    para("We trained the projection, fusion and head in NumPy on CPU using AdamW (learning rate "
         "0.001, weight decay 0.01), 20 epochs, and seeds 42, 43 and 44. For each epoch, we independently "
         "shuffled each class, split it into 67 chunks, paired the chunks, and shuffled within batches. "
         "This uses every example once and includes both classes at roughly their original proportions. "
         "Batches contain at most 128 examples.")
    para("For each run we retained the earliest epoch with maximum validation AUROC. Positive OP "
         "weights 0.1, 0.3 and 1.0 were compared by mean validation AUROC. The zero-weight gated "
         "ablation uses the same settings, initialization, batch order and dropout stream per seed. "
         "Official held-out labels are unavailable. After selection, we exported test probabilities "
         "for the highest-mean configuration using the predetermined seed 42.")
    epochs = ["/".join(str(run["epoch"]) for run in trial["runs"])
              for trial in (linear, gated, op)]
    para(f"Selected epochs for seeds 42/43/44. Weighted addition: {epochs[0]}. "
         f"Gated CE: {epochs[1]}. Gated CE+OP: {epochs[2]}.")
    labels = [["Cached CLIP + MPNet", "→", "128-d projections", "→", "Fusion", "→",
               "Two-logit head", "→", "Softmax class 1"]]
    pipeline = Table(labels, hAlign="CENTER", style=[
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ] + [("BOX", (i, j), (i, j), 0.5, colors.black)
         for i in range(0, 9, 2) for j in (0,)])
    pipeline.spaceBefore = 6
    pipeline.spaceAfter = 3
    story.append(pipeline)
    para('<para alignment="center"><i>Figure 1:</i> Multimodal classification pipeline.</para>',
         "Caption")
    story.append(PageBreak())

    para("4. Results and comparison", "Section")
    para("AUROC uses continuous hateful-class scores. Values are validation means and sample "
         "standard deviations across seeds 42, 43 and 44.")
    data = [["Model", "Mean AUROC", "SD", "Trainable"]]
    model_rows = [
        ("CLIP image only (A1)", image, 513),
        ("MPNet text only (A1)", text, 769),
        ("Weighted addition + CE", linear, linear["runs"][0]["parameters"]),
        ("Gated fusion + CE", gated, gated["runs"][0]["parameters"]),
        (f"Gated fusion + CE + OP ({op['lambda']:g})", op, op["runs"][0]["parameters"]),
    ]
    for name, m, params in model_rows:
        data.append([name, f"{m['mean']:.4f}", f"{m['std']:.4f}", f"{params:,}"])
    para("<i>Table 1:</i> Validation AUROC. A1 marks reused baselines.", "Caption")
    table(data, [80, 34, 22, 36], highlight=4)
    para(f"Gated CE improved over CLIP by <b>{gated['mean']-image['mean']:+.4f}</b> AUROC and "
         f"over weighted addition by <b>{gated['mean']-linear['mean']:+.4f}</b>. Gating beat "
         "addition in all three seeds, though it also adds capacity. The best positive OP weight "
         f"scored <b>{op['mean']-gated['mean']:+.4f}</b> relative to gated CE. Its paired seed "
         "differences were +0.0002, -0.0038 and -0.0039, with magnitudes comparable to the seed SDs. "
         "Weights 0.3 and 1.0 gave means of 0.7224 and 0.7141. These validation results show no "
         "consistent gain from OP. Compact class clusters may obscure distinct targets, visual "
         "contexts and forms of irony within each label.")
    para("5. Six validation examples", "Section")
    para(f"Seed {analysis['seed']}, gated CE+OP (weight {op['lambda']:g}), threshold 0.5. We inspected "
         "the two most confident false positives and false negatives, then two correct fusion "
         "decisions with the largest score shift from either unimodal model.")
    notes = {
        73021: "A shrug accompanies a reference to Jews disappearing. The model may read an implied threat into the caption.",
        34975: "A reaction photo frames a religious parody. The sexual-orientation reference may drive a false alarm.",
        76921: "The caption excuses a threatening gesture toward a woman. Its ordinary wording can hide the implied justification.",
        82301: "A howling dog accompanies an elongated racial slur. The altered spelling and benign scene may hide the cue.",
        19536: "The caption and exaggerated nose invoke an antisemitic stereotype about money. Fusion corrects the image decision.",
        5349: "The photo shows two women sitting by a wall. The country caption supplies no explicit insult. Fusion corrects both decisions.",
    }
    cases = [["ID", "Gold", "CLIP", "MPNet", "FOP", "Inspection"]]
    for case in analysis["cases"]:
        meme_id = case["id"]
        cases.append([f"{meme_id:05d}", str(case["label"]), f"{case['image']:.3f}",
                      f"{case['text']:.3f}", f"{case['fop']:.3f}",
                      Paragraph(notes[meme_id], styles["Cell"])])
    table(cases, [16, 11, 15, 16, 15, 99], alignment="LEFT")
    para("These examples illustrate possible failure modes. They do not estimate their prevalence.")
    para("Group contributions", "Section")
    para("<b>Assigned roles.</b> Bogdan Fanea handled the experimental runs, including feature "
         "alignment, baseline reproduction, ablation training, metric verification, checkpoint "
         "checks and report formatting. Lena Bryll worked on the FOP architecture, loss and "
         "training code. Natalie Stadler worked on qualitative error analysis and report writing.")

    para("Sources and reused implementations", "Section")
    para("We reused Assignment 1 feature caches and baseline scores. The NumPy implementation "
         "adapts the official FOP fusion and loss code [2]. Checkpoints, seed histories and test "
         "predictions accompany the report.")
    para("AI usage", "Section")
    para("We used ChatGPT to help develop and debug the NumPy implementation and to write "
         "checks for feature alignment, checkpoint reproducibility and prediction outputs. "
         "We also used it to draft and revise explanations of the methods and results, and to "
         "adapt the report layout to match the Assignment 1 example. The reported "
         "scores and predictions were computed by running the assignment code.")
    para("References", "ReferenceTitle")
    for ref in (
        '[1] M. S. Saeed et al. Fusion and Orthogonal Projection for Improved Face-Voice Association. '
        'ICASSP, 2022, pp. 7057-7061. <link href="https://arxiv.org/abs/2112.10483">arXiv:2112.10483</link>.',
        '[2] Saeed et al. <link href="https://github.com/msaadsaeed/FOP">Official FOP implementation</link>. '
        'Projection and fusion blocks in retrieval_model.py and OP loss in main.py.',
        '[3] D. Kiela et al. The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes. '
        'NeurIPS, 2020. <link href="https://arxiv.org/abs/2005.04790">arXiv:2005.04790</link>.',
        '[4] OpenAI. <link href="https://huggingface.co/openai/clip-vit-base-patch32">CLIP ViT-B/32 checkpoint</link>. '
        'Revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. The frozen visual projection produces 512 features.',
        '[5] Sentence Transformers. <link href="https://huggingface.co/sentence-transformers/all-mpnet-base-v2">all-mpnet-base-v2 checkpoint</link>. '
        'Revision e8c3b32edf5434bc2275fc9bab85f82640a19130.',
    ):
        para(ref, "Reference")

    def footer(canvas, doc):
        canvas.setFont("Times-Roman", 10)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))

    output = HERE / "output" / "pdf" / "report.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, topMargin=15 * mm,
                            bottomMargin=16 * mm, leftMargin=17 * mm, rightMargin=17 * mm,
                            title="Assignment 2 - Multimodal Hateful Meme Classification",
                            author="Bogdan Fanea, Lena Bryll, Natalie Stadler")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)



if __name__ == "__main__":
    main()
