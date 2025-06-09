
from typing import List


from enum import Enum
class EvalDFColname(Enum):
    readcount = "read_count"
    count_Xmic = "count_assigned_to_Xmic"
    fraction_Xmic = "fraction_assigned_to_Xmic"
    among_signalling_genes = "is_among_signalling_genes"



def parse_arg_evalulate_on_sections(
    dict_all4_configs:dict,
    data_mintflow:dict,
    evalulate_on_sections: List[int] | List[str] | str
) -> List[int]:
    """
    Processes the argument `evalulate_on_sections` and converts it to a list of tissue indices to consider.
    :param dict_all4_configs
    :param data_mintflow:
    :param evalulate_on_sections:
    :return: list_slice_id: a list of integers
    """
    # check the data type of `evalulate_on_sections` ========
    flag_type_correct = False
    if isinstance(evalulate_on_sections, list):
        for u in evalulate_on_sections:
            if isinstance(u, int):
                flag_type_correct = True

    if isinstance(evalulate_on_sections, list):
        for u in evalulate_on_sections:
            if isinstance(u, str):
                flag_type_correct = True

    if isinstance(evalulate_on_sections, str):
        flag_type_correct = True

    if not flag_type_correct:
        raise Exception(
            "The data type of the provdide `evalulate_on_sections` is incorrect." +\
            "Please refer to the documentation of `evaluate_by_DB_signalling_genes` for the correct format."
        )

    if isinstance(evalulate_on_sections, list) and isinstance(evalulate_on_sections[0], int):  # case 1
        return evalulate_on_sections
    elif isinstance(evalulate_on_sections, list) and isinstance(evalulate_on_sections[0], str):  # case 2
        list_sliceidx_toret = []
        for idx_sl, sl in enumerate(data_mintflow['evaluation_list_tissue_section'].list_slice):
            if list(
                set(
                    sl.adata.obs[
                        dict_all4_configs['config_data_evaluation'][idx_sl]['obskey_sliceid_to_checkUnique']
                    ]
                )
            )[0] in evalulate_on_sections:
                list_sliceidx_toret.append(idx_sl)

        return list_sliceidx_toret

    elif isinstance(evalulate_on_sections, str):  # case 3
        if evalulate_on_sections != 'all':
            raise Exception(
                "When the passed `evalulate_on_sections` is a string, the only valid value is 'all' (i.e. evaluation will be done on all tisseu sections.)"
            )

        return list(
            range(
                len(
                    data_mintflow['evaluation_list_tissue_section'].list_slice
                )
            )
        )


