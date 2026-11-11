"""Step 6 - One-page decision report (Arabic, Markdown) built from the analysis results."""
import json

from .config import CONFIG, RESULTS_DIR


def _fmt_ci(r, k=1.0, d=2, unit=""):
    return f"{r['diff'] * k:+.{d}f}{unit} (95% CI: {r['ci_low'] * k:+.{d}f} إلى {r['ci_high'] * k:+.{d}f})"


def run(power: dict, quality: dict, res: dict) -> str:
    c = CONFIG
    p, pc, reg = res["primary"]["unadjusted"], res["primary"]["cuped"], res["primary"]["regression"]
    sec, g, dec = res["secondary"], res["guardrails"], res["decision"]
    nov, seg = res["novelty"], res["segments"]
    s = quality["srm"]
    with open(RESULTS_DIR / "simulation_truth.json", encoding="utf-8") as f:
        truth = json.load(f)

    sec_names = {"recommendation_ctr": "معدل النقر على التوصيات (CTR)",
                 "items_viewed_per_session": "عدد العناصر المشاهدة لكل جلسة",
                 "sessions_per_user_7d": "عدد الجلسات لكل مستخدم (7 أيام)"}
    sec_rows = "\n".join(
        f"| {sec_names[k]} | {v['mean_A']:.4g} | {v['mean_B']:.4g} | {v['rel_lift']:+.1%} | "
        f"{v['p_value']:.3g} | {v['p_value_bh']:.3g} | {'دال' if v['significant_after_bh'] else 'غير دال'} |"
        for k, v in sec.items())

    ch, lat = g["churn_rate_14d"], g["avg_latency_ms"]
    max_smd = max(abs(v) for v in quality["balance_smd"].values())
    w1, w2 = nov["week1"]["diff"], nov["week2"]["diff"]
    if w2 >= 0.8 * w1:
        novelty_text = "الأثر لم يتراجع في الأسبوع الثاني، فلا يوجد دليل على أن الأثر مؤقت (أثر حداثة)"
    else:
        novelty_text = ("الأثر تراجع في الأسبوع الثاني، وهذا مؤشر على أثر حداثة؛ "
                        "نعتمد تقدير الأسبوع الثاني كتقدير متحفظ للأثر طويل المدى")
    report = f"""<div dir="rtl">

# تقرير التجربة: {c['project_ar']}

> **البيانات محاكاة** ببذرة ثابتة ({c['seed']})، والأثر الحقيقي المحدد في المحاكاة: +{truth['true_lift_rel']:.0%}.

## القرار: **{dec['decision']}**
{dec['decision_ar']}

## المقياس الرئيسي: متوسط وقت الجلسة لكل مستخدم (دقيقة)

| البند | القيمة |
|---|---|
| المجموعة الضابطة A | {p['mean_A']:.2f} دقيقة (n = {p['n_A']:,}) |
| المجموعة المعالجة B | {p['mean_B']:.2f} دقيقة (n = {p['n_B']:,}) |
| الأثر (B − A) | {_fmt_ci(p, unit=' دقيقة')} ، p = {p['p_value']:.2g} |
| الأثر النسبي | {p['rel_lift']:+.1%} (من {p['rel_ci_low']:+.1%} إلى {p['rel_ci_high']:+.1%}) |
| الأثر مع CUPED | {_fmt_ci(pc, unit=' دقيقة')} ، انخفاض التباين {res['primary']['cuped_variance_reduction']:.0%} |
| الانحدار مع المتغيرات المساعدة | {_fmt_ci(reg, unit=' دقيقة')} |
| اختبار Mann-Whitney (للتحقق من الالتواء) | p = {res['primary']['mann_whitney_p']:.2g} |
| أصغر أثر مهم (MDE) | {dec['mde_abs_min']:.2f} دقيقة (+{c['mde_rel']:.0%}) |

## المقاييس الثانوية (تصحيح Benjamini-Hochberg)

| المقياس | A | B | الأثر النسبي | p | p بعد التصحيح | الحالة |
|---|---|---|---|---|---|---|
{sec_rows}

## مقاييس الحماية (عدم التدهور)

| المقياس | A | B | الفرق | الحد المسموح | الحالة |
|---|---|---|---|---|---|
| معدل إلغاء الاشتراك خلال 14 يومًا | {ch['mean_A']:.2%} | {ch['mean_B']:.2%} | {_fmt_ci(ch, 100, 2, ' نقطة')} | +{c['churn_margin_abs'] * 100:.1f} نقطة | {'سليم' if ch['ok'] else 'تدهور'} |
| متوسط زمن استجابة الخادم | {lat['mean_A']:.1f} ms | {lat['mean_B']:.1f} ms | {_fmt_ci(lat, 1, 1, ' ms')} | +{c['latency_margin_ms']:.0f} ms | {'سليم' if lat['ok'] else 'تدهور'} |

## فحوص الثقة
- SRM: A = {s['n_A']:,} و B = {s['n_B']:,}، χ² = {s['chi2']:.2f}، p = {s['p_value']:.2f} → {'خلل' if s['srm_detected'] else 'التوزيع سليم'}.
- حُذف {quality['exact_duplicate_rows_removed']} سجلًا مكررًا، وقيم زمن الاستجابة المفقودة ({quality['missing_values'].get('avg_latency_ms', 0)}) استُبعدت من مقياس الحماية فقط.
- توازن المتغيرات قبل التجربة: أكبر قيمة |SMD| = {max_smd:.3f} ({'أقل من 0.1، المجموعتان متوازنتان' if max_smd < 0.1 else 'أعلى من 0.1، يوجد اختلال'}).

## ملاحظات إضافية
- أثر الحداثة: الأثر في الأسبوع الأول {nov['week1']['diff']:+.2f} دقيقة مقابل {nov['week2']['diff']:+.2f} في الأسبوع الثاني؛ {novelty_text}.
- الأجهزة (استكشافي): الجوال {seg['by_device']['mobile']['rel_lift']:+.1%} والويب {seg['by_device']['web']['rel_lift']:+.1%}، واختبار التفاعل p = {seg['interaction_p_value']:.2f}.
- الفرضية توقعت +15%، والأثر المقدر {p['rel_lift']:+.1%}؛ {'وقيمة 15% داخل فترة الثقة' if dec['hypothesized_15pct_inside_ci'] else 'و15% خارج فترة الثقة، أي أن الأثر حقيقي لكنه أصغر من المتوقع'}.
- الأثر التجاري التقريبي: {dec['extra_min_per_user_first_week']:.1f} دقيقة مشاهدة إضافية لكل مستخدم في أسبوعه الأول، أي نحو {dec['extra_hours_first_week_all_annual_users']:,.0f} ساعة سنويًا لـ {c['annual_users']:,} مستخدم جديد.

</div>
"""
    path = RESULTS_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"  Report written -> outputs/results/{path.name}")
    return report
