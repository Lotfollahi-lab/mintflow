
from typing import List

def parse_arg_evalulate_on_sections(
    data_mintflow:dict,
    evalulate_on_sections: List[int] | List[str] | str
) -> List[int]:
    """
    Processes the argument `evalulate_on_sections` and converts it to a list of tissue indices to consider.
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
        pass
        # TODO:HERE complete the implementation
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


