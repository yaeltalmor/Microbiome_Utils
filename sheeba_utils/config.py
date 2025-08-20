from enum import Enum

#---------------------COLUMNS CATEGORIES------------------#
EX_MNFST_CAT_COLS = [
    'extraintestinal_manifestations_organs_2',
    'extraintestinal_manifestations_organs_3',
    'extraintestinal_manifestations_organs_last'
]

EX_MNFST_NON_IBD_COLS = [
    'non_ibd_manifestations_2_date',
    'non_ibd_manifestations_3_date',
    'non_ibd_manifestations_last_date'
]

BIOLOGIC_DRUG_COLUMNS = [
    "drug1_date", "drug2_date", "drug3_date", "drug4_date",
    "drug5_date", "drug6_date", "drug7_date", "drug8_date"
]

SMALL_MOLECULE_DRUG_COLUMNS = [
    'five_asa_start_date', 'calcineurin_inhibitor_start_date', 'corticos_start_date',
    'mtx_start_date', 'thiopurines_start_date'
]

DURATION_COLUMNS = [
    "drug1_duration", "drug2_duration", "drug3_duration", "drug4_duration",
    "drug5_duration", "drug6_duration", "drug7_duration", "drug8_duration"
]

PROCEDURE_COLUMNS = [
    "procedure_date_1", "procedure_date_2", "last_procedure_date"
]

IBD_RELATED_MNFSTS = [
    'extraintestinal_manifestations_2_date',
    'extraintestinal_manifestations_3_date',
    'extraintestinal_manifestations_last_date'
]

NEOPLASMS_COLUMNS = [
    'neoplasms_date_2', 'neoplasms_date_3', 'last_date_neoplasms'
]

NON_IBD_MNFSTS = EX_MNFST_NON_IBD_COLS + NEOPLASMS_COLUMNS

MED_TESTS_COLUMNS = [
    'weight_date', 'main_crp_date_2', 'main_crp_date_3', 'last_date_of_crp',
    'uc.sccai_summary_2_date', 'uc.sccai_summary_3_date', 'uc.sccai_summary_last_date',
    'calprotectin_date_2', 'calprotectin_date_3', 'last_result_date_calprotectin',
    'visits_gastro_endoscopy_procedures_date', 'colonoscopy_answer_fileds_date'
]

GASTRO_VISITS_COLUMNS = [
    'first_visit_gastro_date', 'visits_gastro_clinic_date',
    'visits_gastro_hosp_date', 'last_visit_er_date'
]

TRAJECTORY_COLUMNS = ['end_of_data']

CRP_COLUMNS = ['main_crp_2', 'main_crp_3', 'last_crp']
CRP_DATE_COLUMNS = ['main_crp_date_2', 'main_crp_date_3', 'last_date_of_crp']

#---------------------DATE EVENT MAPPERS------------------#
GENERAL_DATE_EVENT_MAPPING = {
     # life events
    'birth_date':('life_event', 'birth'),
    'death_date':('life_event','death'),
    'diagnosis_date':('life_event','diagnosis'),
     # non biological treatments
    'five_asa_start_date':('small_drug_subscription','five_asa'),
    'calcineurin_inhibitor_start_date':('small_drug_subscription','calcineurin_inhibitor'),
    'corticos_start_date':('small_drug_subscription','corticos'),
    'mtx_start_date':('small_drug_subscription','mtx'),
    'thiopurines_start_date': ('small_drug_subscription','thiopurines'),
    # medical exams events
    'visits_gastro_endoscopy_procedures_date':('medical_exam__endoscopy','endoscopy'),
    'colonoscopy_answer_fileds_date':('medical_exam__colonoscopy','colonoscopy'),
    # events that might be relevant
    'first_visit_gastro_date':('sheeba_visit','gastro_visit'),
    'visits_gastro_clinic_date':('sheeba_visit','gastro_clinic'),
    'visits_gastro_hosp_date':('sheeba_visit','gastro_hospital'),
    'last_visit_er_date':('sheeba_visit','er_visit'),
    'last_visit_in_child_department_date': ('sheeba_visit','last_visit_in_child_department'),
    # Extra-intestinal Manifestations
    'non_ibd_manifestations_2_date': (
    'extraintestinal_manifestations__not_ibd_related', 'non_ibd_manifestation'),
    'non_ibd_manifestations_3_date': (
    'extraintestinal_manifestations__not_ibd_related', 'non_ibd_manifestation'),
    'non_ibd_manifestations_last_date': (
    'extraintestinal_manifestations__not_ibd_related', 'non_ibd_manifestation'),
}
PER_PATIENT_DATE_EVENT_MAPPING = {
    # biological treatments
    'drug1_date':('bio_drug_subscription','drug1'),
    'drug2_date':('bio_drug_subscription','drug2'),
    'drug3_date':('bio_drug_subscription','drug3'),
    'drug4_date':('bio_drug_subscription','drug4'),
    'drug5_date':('bio_drug_subscription','drug5'),
    'drug6_date':('bio_drug_subscription','drug6'),
    'drug7_date':('bio_drug_subscription','drug7'),
    'drug8_date':('bio_drug_subscription','drug8'),
    # medical exams events
    'weight_date':('medical_exam__bmi','bmi'),
    'main_crp_date_2':('medical_exam__crp','main_crp_2'),
    'main_crp_date_3':('medical_exam__crp','main_crp_3'),
    'last_date_of_crp':('medical_exam__crp','last_crp'),
    'uc.sccai_summary_2_date':('medical_exam__sccai','uc.sccai_summary_2'),
    'uc.sccai_summary_3_date':('medical_exam__sccai','uc.sccai_summary_3'),
    'uc.sccai_summary_last_date':('medical_exam__sccai','uc.sccai_summary_last'),
    'calprotectin_date_2':('medical_exam__calprotectin','calprotectin_result_2'),
    'calprotectin_date_3':('medical_exam__calprotectin','calprotectin_result_3'),
    'last_result_date_calprotectin':('medical_exam__calprotectin','last_result_calprotectin'),
    # non-pharmacological intervention
    'procedure_date_1':('procedure','procedure_1_category'),
    'procedure_date_2':('procedure','procedure_2_category'),
    'last_procedure_date':('procedure','last_procedure_category'),
    # external manifestation
    'extraintestinal_manifestations_2_date':('extraintestinal_manifestations__ibd_related','extraintestinal_manifestations_organs_2'),
    'extraintestinal_manifestations_3_date':('extraintestinal_manifestations__ibd_related','extraintestinal_manifestations_organs_3'),
    'extraintestinal_manifestations_last_date':('extraintestinal_manifestations__ibd_related','extraintestinal_manifestations_organs_last'),
    'neoplasms_date_2':('extraintestinal_manifestations__not_ibd_related','neoplasms_diagnosis_2_category'),
    'neoplasms_date_3':('extraintestinal_manifestations__not_ibd_related','neoplasms_diagnosis_3_category'),
    'last_date_neoplasms':('extraintestinal_manifestations__not_ibd_related','last_neoplasms_diagnosis_category'),
}

# Medical exams mappers
CRP_DATE_VALUE_MAPPING = {
    'main_crp_date_2': 'main_crp_2',
    'main_crp_date_3': 'main_crp_3',
    'last_date_of_crp': 'last_crp'
}
SCCAI_DATE_VALUE_MAPPING = {
    'uc.sccai_summary_2_date':'uc.sccai_summary_2',
    'uc.sccai_summary_3_date':'uc.sccai_summary_3',
    'uc.sccai_summary_last_date':'uc.sccai_summary_last',
}
CALPROTECTIN_DATE_VALUE_MAPPING = {
    'calprotectin_date_2':'calprotectin_result_2',
    'calprotectin_date_3':'calprotectin_result_3',
    'last_result_date_calprotectin':'last_result_calprotectin',
}

MED_EXAMS_MAPPER = {
    'crp': CRP_DATE_VALUE_MAPPING,
    'sccai_summary': SCCAI_DATE_VALUE_MAPPING,
    'calprotectin': CALPROTECTIN_DATE_VALUE_MAPPING,
}

#---------------------GENERAL DEFINITIONS------------------#
class MultiHotMethod(Enum):
    ONE_HOT = "one_hot"
    ORDER_HOT = "order_hot"
    MONTHS_SINCE_HOT = "months_since_hot"
