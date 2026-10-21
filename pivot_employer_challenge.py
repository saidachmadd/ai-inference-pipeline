"""
5-MINUTE ARCHITECTURE PIVOT CHALLENGE (STRATEGY PATTERN DEMO)
Pivot Domain: Dari Multimodal Medical Payload -> Corporate Payroll Distribution & Entity Reconciliation
Jalankan langsung: python pivot_employer_challenge.py
"""

import json
from typing import Any

from pydantic import BaseModel, Field


class EmployerCompliancePayload(BaseModel):
    company_id: str = Field(..., examples=["CORP-3216-IND88"])
    industry_sector: str = Field(..., examples=["MANUFACTURING_AUTOMOTIVE"])
    total_active_employees_tax_record: int = Field(..., gt=0)
    bpjs_registered_employees: int = Field(..., gt=0)
    senior_And_managerial_staff_count: int = Field(..., ge=0)
    employees_reported_at_exact_umk: int = Field(..., ge=0)
    avg_reported_monthly_wage_idr: float = Field(..., gt=0)
    sectoral_benchmark_monthly_wage_idr: float = Field(..., gt=0)
    regional_umk_idr: float = Field(..., gt=0)


class EmployerWageFeatureExtractor:
    """
    Pluggable Feature Extractor untuk mendeteksi distorsi distribusi statistik gaji
    (artificial floor clustering & benchmark deviation) serta selisih entitas lintas tabel.
    """

    @staticmethod
    def transform(payload: EmployerCompliancePayload) -> dict[str, Any]:
        # 1. Benchmark Deviation Ratio (< 0.55 mengindikasikan distribusi gaji ditekan secara sistematis)
        wage_ratio = round(
            payload.avg_reported_monthly_wage_idr
            / payload.sectoral_benchmark_monthly_wage_idr,
            4,
        )

        # 2. Artificial Floor Clustering Ratio (proporsi entitas yang menumpuk tepat di batas bawah/UMK)
        umk_clustering = round(
            payload.employees_reported_at_exact_umk
            / max(payload.bpjs_registered_employees, 1),
            4,
        )

        # 3. Cross-Database Entity Gap Ratio (selisih entitas aktif di log pajak vs log terdaftar)
        headcount_gap = round(
            (
                payload.total_active_employees_tax_record
                - payload.bpjs_registered_employees
            )
            / max(payload.total_active_employees_tax_record, 1),
            4,
        )

        # Kalkulasi ekspektasi nilai finansial yang terdampak per bulan (5% rate)
        wage_suppression_per_worker = max(
            0.0,
            payload.sectoral_benchmark_monthly_wage_idr
            - payload.avg_reported_monthly_wage_idr,
        )
        estimated_monthly_contribution_loss_idr = round(
            (wage_suppression_per_worker * payload.bpjs_registered_employees * 0.05)
            + (
                headcount_gap
                * payload.total_active_employees_tax_record
                * payload.sectoral_benchmark_monthly_wage_idr
                * 0.05
            ),
            2,
        )

        discrepancies = []
        if wage_ratio < 0.55:
            discrepancies.append(
                f"Statistical distribution skew (Under-Reporting): mean gaji terlapor (Rp{payload.avg_reported_monthly_wage_idr:,.0f}) "
                f"terdeviasi jauh menjadi hanya {wage_ratio:.1%} dari benchmark sektoral {payload.industry_sector} (Rp{payload.sectoral_benchmark_monthly_wage_idr:,.0f})."
            )
        if umk_clustering > 0.80 and payload.senior_And_managerial_staff_count > 5:
            discrepancies.append(
                f"Artificial floor clustering: {umk_clustering:.1%} entitas terkonsentrasi tepat pada nilai batas bawah "
                f"(Rp{payload.regional_umk_idr:,.0f}), kontradiktif terhadap keberadaan {payload.senior_And_managerial_staff_count} entitas berprofil senior/manajerial."
            )
        if headcount_gap > 0.15:
            discrepancies.append(
                f"Cross-database entity mismatch (Partial Registration): {headcount_gap:.1%} entitas aktif pada catatan pajak tidak ditemukan pada tabel kepesertaan."
            )

        return {
            "wage_to_sectoral_benchmark_ratio": wage_ratio,
            "umk_clustering_ratio": umk_clustering,
            "headcount_gap_ratio": headcount_gap,
            "estimated_monthly_contribution_loss_idr": estimated_monthly_contribution_loss_idr,
            "discrepancies": discrepancies,
        }


if __name__ == "__main__":
    sample_employer = EmployerCompliancePayload(
        company_id="CORP-3216-IND88",
        industry_sector="MANUFACTURING_AUTOMOTIVE",
        total_active_employees_tax_record=250,
        bpjs_registered_employees=180,
        senior_And_managerial_staff_count=25,
        employees_reported_at_exact_umk=165,
        avg_reported_monthly_wage_idr=5300000.0,
        sectoral_benchmark_monthly_wage_idr=10500000.0,
        regional_umk_idr=5250000.0,
    )
    result = EmployerWageFeatureExtractor.transform(sample_employer)
    print(json.dumps(result, indent=2, ensure_ascii=False))
