"""
shaft_analyser/report_generator.py
-----------------------------------
Generates formatted console and file reports from AnalysisResult objects.
"""

from datetime import datetime
from pathlib import Path
from models import AnalysisResult, ShaftProfile


RISK_ICONS = {
    "LOW":      "✓",
    "MEDIUM":   "⚠",
    "HIGH":     "✖",
    "CRITICAL": "✖✖",
}

RISK_COLORS = {
    "LOW":      "\033[92m",   # green
    "MEDIUM":   "\033[93m",   # yellow
    "HIGH":     "\033[91m",   # red
    "CRITICAL": "\033[95m",   # magenta
}
RESET = "\033[0m"
BOLD  = "\033[1m"


class ReportGenerator:
    """
    Renders AnalysisResult to console (coloured) and/or plain-text file.

    Usage
    -----
        gen = ReportGenerator(output_dir="reports")
        gen.print_console(shaft, result)
        gen.save_to_file(shaft, result)
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)

    def print_console(self, shaft: ShaftProfile, result: AnalysisResult):
        color = RISK_COLORS.get(result.fatigue_risk, "")
        icon  = RISK_ICONS.get(result.fatigue_risk, "?")

        print()
        print(f"{BOLD}{'═' * 58}{RESET}")
        print(f"{BOLD}  SHAFT FAILURE RISK REPORT — {result.shaft_id}{RESET}")
        print(f"{'═' * 58}")

        print(f"\n  {'Material':<26} {shaft.material}")
        print(f"  {'Diameter × Length':<26} {shaft.diameter_mm} mm × {shaft.length_mm} mm")
        print(f"  {'Operating speed':<26} {shaft.rpm:.0f} RPM")
        print(f"  {'Transmitted power':<26} {shaft.power_kw:.1f} kW")
        print(f"  {'Bearing span':<26} {shaft.bearing_span_mm} mm")
        print(f"  {'Keyway present':<26} {'Yes' if shaft.keyway else 'No'}")
        print(f"  {'Surface finish Ra':<26} {shaft.surface_finish_ra} µm")

        print(f"\n  {'─' * 54}")
        print(f"  {'CALCULATED STRESSES':}")
        print(f"  {'─' * 54}")
        print(f"  {'Torque':<26} {result.torque_nm:>8.1f} N·m")
        print(f"  {'Torsional shear stress':<26} {result.shear_stress_mpa:>8.2f} MPa")
        print(f"  {'Bending stress':<26} {result.bending_stress_mpa:>8.2f} MPa")
        print(f"  {'Von Mises stress':<26} {result.von_mises_mpa:>8.2f} MPa")
        print(f"  {'Yield strength':<26} {shaft.yield_strength:>8} MPa")
        print(f"  {'Safety factor':<26} {result.safety_factor:>8.2f}x")

        print(f"\n  {'─' * 54}")
        sf_color = "\033[92m" if result.safety_factor >= 1.5 else "\033[91m"
        print(f"  Risk score   : {color}{BOLD}{result.risk_score:.1f} / 100{RESET}")
        print(f"  Risk level   : {color}{BOLD}{icon}  {result.fatigue_risk}{RESET}")
        print(f"  Safety factor: {sf_color}{BOLD}{result.safety_factor:.2f}x{RESET}"
              f"  (recommended ≥ 1.5)")

        print(f"\n  {'─' * 54}")
        print(f"  FLAGS:")
        for f in result.flags:
            print(f"    • {f}")

        print(f"\n  RECOMMENDATIONS:")
        for r in result.recommendations:
            print(f"    → {r}")

        print(f"\n{'═' * 58}\n")

    def save_to_file(self, shaft: ShaftProfile, result: AnalysisResult) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"report_{result.shaft_id}_{ts}.txt"

        lines = [
            "=" * 58,
            f"  SHAFT FAILURE RISK REPORT — {result.shaft_id}",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 58,
            "",
            "INPUT PARAMETERS",
            "-" * 40,
            f"  Material           : {shaft.material}",
            f"  Diameter           : {shaft.diameter_mm} mm",
            f"  Length             : {shaft.length_mm} mm",
            f"  RPM                : {shaft.rpm:.0f}",
            f"  Power              : {shaft.power_kw:.1f} kW",
            f"  Bearing span       : {shaft.bearing_span_mm} mm",
            f"  Keyway             : {'Yes' if shaft.keyway else 'No'}",
            f"  Surface finish Ra  : {shaft.surface_finish_ra} µm",
            "",
            "STRESS ANALYSIS",
            "-" * 40,
            f"  Torque             : {result.torque_nm:.1f} N·m",
            f"  Shear stress       : {result.shear_stress_mpa:.2f} MPa",
            f"  Bending stress     : {result.bending_stress_mpa:.2f} MPa",
            f"  Von Mises stress   : {result.von_mises_mpa:.2f} MPa",
            f"  Yield strength     : {shaft.yield_strength} MPa",
            f"  Safety factor      : {result.safety_factor:.2f}x",
            "",
            "RISK ASSESSMENT",
            "-" * 40,
            f"  Risk score         : {result.risk_score:.1f} / 100",
            f"  Risk level         : {result.fatigue_risk}",
            "",
            "FLAGS",
            "-" * 40,
        ]
        for f in result.flags:
            lines.append(f"  • {f}")
        lines += ["", "RECOMMENDATIONS", "-" * 40]
        for r in result.recommendations:
            lines.append(f"  → {r}")
        lines += ["", "=" * 58]

        with open(filepath, "w") as fh:
            fh.write("\n".join(lines))

        return filepath
