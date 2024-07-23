# flake8: noqa: B950

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
    CHOICE_COL,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import LearnedHeuristic


class PadMMA100(LearnedHeuristic):
    def __init__(self) -> None:
        pass

    def check_precondition(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == 166912
            and str(metadata.device_capa) == "(8, 0)"
        )

    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        context.context_dict[CHOICE_COL] = choice
        return self.predict(context)

    def get_speedup_threshold(self) -> float:
        return 1.7025303314066

    def get_name(self) -> str:
        return "pad_mm"

    def predict(self, context: AHContext) -> float:
        if str(context.get_value("choice")) != "pad":
            if str(context.get_value("using_tf32")) != "False":
                if context.get_value("m*n") <= 4171264.0:
                    if context.get_value("m*k") <= 3999308.0:
                        return 1.8751469764071178
                    else:
                        if str(context.get_value("n_multiple_32")) != "True":
                            return 0.9117231355626345
                        else:
                            return 1.1607689608873861
                else:
                    if str(context.get_value("n_multiple_2")) != "True":
                        if str(context.get_value("using_tf32")) != "True":
                            return 0.7430382200435992
                        else:
                            return 0.8531269794448678
                    else:
                        if str(context.get_value("k_multiple_2")) != "True":
                            return 0.7577181972719917
                        else:
                            return 0.8977349440424219
            else:
                if context.get_value("m*n") <= 1299712.0:
                    return 1.1669723418995592
                else:
                    if context.get_value("mat2_stride_1") <= 45217.5:
                        if context.get_value("m*n") <= 55884158.0:
                            return 1.0262769936909601
                        else:
                            return 1.0022677428470845
                    else:
                        if context.get_value("m") <= 18478.0:
                            return 1.1127066261894312
                        else:
                            return 1.0337740659894263
        else:
            if str(context.get_value("mat1_dtype")) != "torch.float32":
                if str(context.get_value("n_multiple_2")) != "False":
                    if str(context.get_value("k_multiple_2")) != "True":
                        if context.get_value("mat1_stride_0") <= 561.0:
                            return 1.2900382135142956
                        else:
                            return 1.5761737616057887
                    else:
                        if context.get_value("num_dims_needs_padding") <= 1.5:
                            return 1.0472263310239422
                        else:
                            return 1.1727673465762514
                else:
                    if context.get_value("k") <= 28238.5:
                        if context.get_value("k/(m*n)") <= 0.00026227018679492176:
                            return 1.6770542505397175
                        else:
                            return 1.3974785435105923
                    else:
                        if str(context.get_value("mat1_dtype")) != "torch.bfloat16":
                            return 1.3952699800111992
                        else:
                            return 1.5759286511628336
            else:
                if str(context.get_value("using_tf32")) != "False":
                    if context.get_value("m*n") <= 14119424.0:
                        return 0.8875772670422478
                    else:
                        if (
                            str(context.get_value("mat2_innermost_needs_padding"))
                            != "True"
                        ):
                            return 1.1467728924377265
                        else:
                            return 1.215842963532998
                else:
                    if context.get_value("arith_intensity") <= 396.8774871826172:
                        return 0.89940161869551
                    else:
                        if context.get_value("mat2_stride_1") <= 45217.5:
                            return 0.9964328169353532
                        else:
                            return 0.9493479238294826
