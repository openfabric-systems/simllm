#include "simllm/rnic/rnic_anomaly_table.h"

namespace simllm::rnic {

const char* toString(AnomalyKind kind) noexcept {
    switch (kind) {
    case AnomalyKind::Emergent:
        return "emergent";
    case AnomalyKind::Injected:
        return "injected";
    case AnomalyKind::Fabric:
        return "fabric";
    case AnomalyKind::Counter:
        return "counter";
    }
    return "invalid";
}

std::string renderRnicAnomalyRow(const RnicAnomalyRow& row) {
    std::string out = "| ";
    out += row.id;
    out += " | ";
    out += row.name;
    out += " | ";
    out += row.trigger;
    out += " | ";
    out += row.effect;
    out += " | ";
    out += row.kind_text;
    out += " | ";
    out += row.evidence;
    out += " |";
    return out;
}

std::string renderRnicAnomalyTableMarkdown() {
    std::string out =
        "# RNIC golden model anomaly table\n"
        "\n"
        "This file is generated from the `constexpr` table in\n"
        "`simllm/backends/rnic/include/simllm/rnic/rnic_anomaly_table.h` and a\n"
        "native test compares it byte for byte. Edit the table, not this file.\n"
        "It is the projection of the anomaly table in\n"
        "[the golden-model design](rnic-cmodel.md), which carries the same\n"
        "rows and states how each one is reproduced.\n"
        "\n"
        "Kinds: `emergent` falls out of a modelled mechanism and is validated,\n"
        "`injected` is applied by rule because the mechanism is not public,\n"
        "`fabric` is a property of the switch or link reproduced by the packet\n"
        "simulator rather than by the endpoint, and `counter` is a facade\n"
        "behaviour with no datapath effect.\n"
        "\n"
        "| id | anomaly | trigger | effect and magnitude | kind | evidence |\n"
        "|---|---|---|---|---|---|\n";
    for (const RnicAnomalyRow& row : kRnicAnomalyTable) {
        out += renderRnicAnomalyRow(row);
        out += "\n";
    }
    out +=
        "\n"
        "Every row is registered against a model block or is explicitly the\n"
        "fabric's. A row whose kind is `emergent` must be reproduced by the\n"
        "named mechanism inside its registered band before that block can be\n"
        "called validated; a row whose kind is `injected` is reproduced by\n"
        "rule and is honest about it.\n";
    return out;
}

}  // namespace simllm::rnic
