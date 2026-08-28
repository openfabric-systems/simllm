// Corrected ordered-pair-class adapter for pinned htsim rnic-nn transports.

#include "rnic_fluid_manifold.h"
#include "rnic_max_min_allocator.h"
#include "rnic_packet_extent.h"
#include "rnic_packetized_manifold.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef HTSIM_SOURCE_COMMIT
#error "HTSIM_SOURCE_COMMIT is required"
#endif

namespace {

using FlowId = RnicMaxMinFlow::FlowId;
using NodeId = RnicMaxMinFlow::NodeId;
using RateBps = RnicMaxMinFlow::RateBps;
using TimePs = std::uint64_t;
using ClassKey = std::pair<NodeId, NodeId>;

struct Options {
    std::string schedule_csv;
    std::string completion_csv;
    std::string manifest_json;
    RateBps source_capacity_bps{0};
    RateBps destination_capacity_bps{0};
    std::uint64_t max_wire_bytes{0};
    std::uint64_t header_bytes{0};
    TimePs propagation_ps{0};
    std::uint32_t node_count{0};
    bool provenance{false};
};

struct FlowRow {
    FlowId numeric_id;
    std::string flow_id;
    std::uint32_t wave;
    NodeId source;
    NodeId destination;
    std::uint64_t payload_bytes;
    TimePs released_at_ps;
};

struct CompletionRow {
    std::string transport;
    FlowRow flow;
    TimePs admitted_at_ps;
    TimePs completion_time_ps;
    std::uint64_t packet_count;
    std::uint64_t wire_bytes;
};

struct SimulationLedger {
    std::uint64_t flow_count{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t packet_count{0};
    std::uint64_t wire_bytes{0};
    std::uint64_t allocation_epochs{0};
    std::uint64_t max_active_classes{0};
    RateBps max_source_allocated_bps{0};
    RateBps max_destination_allocated_bps{0};
    std::uint64_t active_pair_limit_violations{0};
};

std::uint64_t parseUnsigned(const std::string& name, const std::string& value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(name + " must be an unsigned integer");
    }
    std::size_t used = 0;
    const std::uint64_t parsed = std::stoull(value, &used, 10);
    if (used != value.size()) {
        throw std::invalid_argument(name + " must be an unsigned integer");
    }
    return parsed;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--provenance") {
            options.provenance = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(option + " requires a value");
        }
        const std::string value = argv[++index];
        if (option == "--schedule-csv") {
            options.schedule_csv = value;
        } else if (option == "--completion-csv") {
            options.completion_csv = value;
        } else if (option == "--manifest-json") {
            options.manifest_json = value;
        } else if (option == "--source-capacity-bps") {
            options.source_capacity_bps = parseUnsigned(option, value);
        } else if (option == "--destination-capacity-bps") {
            options.destination_capacity_bps = parseUnsigned(option, value);
        } else if (option == "--max-wire-bytes") {
            options.max_wire_bytes = parseUnsigned(option, value);
        } else if (option == "--header-bytes") {
            options.header_bytes = parseUnsigned(option, value);
        } else if (option == "--propagation-ps") {
            options.propagation_ps = parseUnsigned(option, value);
        } else if (option == "--nodes") {
            const std::uint64_t nodes = parseUnsigned(option, value);
            if (nodes > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--nodes exceeds uint32_t");
            }
            options.node_count = static_cast<std::uint32_t>(nodes);
        } else {
            throw std::invalid_argument("unknown option " + option);
        }
    }
    if (options.provenance) {
        return options;
    }
    if (options.schedule_csv.empty() || options.completion_csv.empty()
        || options.manifest_json.empty() || options.source_capacity_bps == 0
        || options.destination_capacity_bps == 0 || options.max_wire_bytes == 0
        || options.node_count == 0) {
        throw std::invalid_argument("all schedule and physical options are required");
    }
    if (options.header_bytes >= options.max_wire_bytes) {
        throw std::invalid_argument("header bytes must be smaller than maximum wire bytes");
    }
    if (options.source_capacity_bps > options.destination_capacity_bps) {
        throw std::invalid_argument(
            "source capacity must not exceed the packet calendar access capacity");
    }
    return options;
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<FlowRow> readSchedule(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open schedule CSV " + path);
    }
    std::string line;
    if (!std::getline(input, line)
        || line
               != "numeric_id,flow_id,wave,source,destination,payload_bytes,released_at_ps") {
        throw std::runtime_error("schedule CSV header is not the frozen schema");
    }
    std::vector<FlowRow> rows;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = splitCsv(line);
        if (fields.size() != 7) {
            throw std::runtime_error("schedule CSV row must have seven fields");
        }
        const std::uint64_t wave = parseUnsigned("wave", fields[2]);
        const std::uint64_t source = parseUnsigned("source", fields[3]);
        const std::uint64_t destination = parseUnsigned("destination", fields[4]);
        if (wave > std::numeric_limits<std::uint32_t>::max()
            || source > std::numeric_limits<std::uint32_t>::max()
            || destination > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("schedule identity exceeds uint32_t");
        }
        rows.push_back(
            {parseUnsigned("numeric_id", fields[0]),
             fields[1],
             static_cast<std::uint32_t>(wave),
             static_cast<NodeId>(source),
             static_cast<NodeId>(destination),
             parseUnsigned("payload_bytes", fields[5]),
             parseUnsigned("released_at_ps", fields[6])});
    }
    if (rows.empty()) {
        throw std::runtime_error("schedule CSV contains no flows");
    }
    std::sort(rows.begin(), rows.end(), [](const FlowRow& lhs, const FlowRow& rhs) {
        return std::tie(lhs.released_at_ps, lhs.numeric_id)
               < std::tie(rhs.released_at_ps, rhs.numeric_id);
    });
    std::set<FlowId> identities;
    for (const FlowRow& row : rows) {
        if (!identities.insert(row.numeric_id).second) {
            throw std::runtime_error("schedule numeric flow IDs are not unique");
        }
    }
    return rows;
}

RnicMaxMinAllocator::CapacityMap sourceCapacities(
        const std::vector<FlowRow>& rows, RateBps capacity_bps) {
    RnicMaxMinAllocator::CapacityMap capacities;
    for (const FlowRow& row : rows) {
        capacities.emplace(row.source, capacity_bps);
    }
    return capacities;
}

RnicMaxMinAllocator::CapacityMap destinationCapacities(
        const std::vector<FlowRow>& rows, RateBps capacity_bps) {
    RnicMaxMinAllocator::CapacityMap capacities;
    for (const FlowRow& row : rows) {
        capacities.emplace(row.destination, capacity_bps);
    }
    return capacities;
}

void recordAllocation(
        const std::vector<RnicMaxMinFlow>& active,
        const RnicMaxMinAllocator::AllocationMap& allocation,
        const Options& options,
        SimulationLedger& ledger) {
    std::map<NodeId, RateBps> by_source;
    std::map<NodeId, RateBps> by_destination;
    std::set<ClassKey> classes;
    for (const RnicMaxMinFlow& flow : active) {
        const RateBps rate = allocation.at(flow.flow_id);
        by_source[flow.source_node] += rate;
        by_destination[flow.destination_node] += rate;
        if (!classes.insert({flow.source_node, flow.destination_node}).second) {
            ++ledger.active_pair_limit_violations;
        }
    }
    ledger.max_active_classes = std::max<std::uint64_t>(
        ledger.max_active_classes, classes.size());
    for (const auto& item : by_source) {
        ledger.max_source_allocated_bps = std::max(
            ledger.max_source_allocated_bps, item.second);
        if (item.second > options.source_capacity_bps) {
            throw std::logic_error("source allocation exceeds corrected capacity");
        }
    }
    for (const auto& item : by_destination) {
        ledger.max_destination_allocated_bps = std::max(
            ledger.max_destination_allocated_bps, item.second);
        if (item.second > options.destination_capacity_bps) {
            throw std::logic_error("destination allocation exceeds corrected capacity");
        }
    }
}

std::vector<CompletionRow> simulatePacket(
        const std::vector<FlowRow>& rows,
        const Options& options,
        SimulationLedger& ledger) {
    struct State {
        FlowRow flow;
        TimePs admitted_at_ps{0};
        std::uint64_t payload_reserved{0};
        std::uint64_t packet_count{0};
        std::uint64_t wire_bytes{0};
        std::optional<TimePs> completion_time_ps;
    };

    const RnicDataPacketizationConfig packetization(
        options.max_wire_bytes, options.header_bytes);
    RnicPacketizedSlotCalendar calendar(
        options.destination_capacity_bps,
        options.max_wire_bytes,
        options.propagation_ps);
    const auto source_capacities = sourceCapacities(rows, options.source_capacity_bps);
    const auto destination_capacities = destinationCapacities(
        rows, options.destination_capacity_bps);
    std::map<ClassKey, std::deque<FlowRow>> waiting;
    std::map<ClassKey, FlowId> active_by_class;
    std::map<FlowId, State> states;
    std::size_t release_index = 0;
    std::uint64_t completed = 0;

    const auto enqueueReleased = [&](TimePs now_ps) {
        while (release_index < rows.size()
               && rows[release_index].released_at_ps <= now_ps) {
            const FlowRow& flow = rows[release_index++];
            waiting[{flow.source, flow.destination}].push_back(flow);
        }
    };
    const auto admitWaiting = [&](TimePs now_ps) {
        for (auto& item : waiting) {
            if (item.second.empty() || active_by_class.count(item.first) != 0) {
                continue;
            }
            const FlowRow flow = item.second.front();
            item.second.pop_front();
            states.emplace(
                flow.numeric_id,
                State{flow, now_ps, 0, 0, 0, std::nullopt});
            active_by_class.emplace(item.first, flow.numeric_id);
        }
    };

    while (completed < rows.size()) {
        if (active_by_class.empty()) {
            calendar.beginEpoch(calendar.nextSlotIndex(), {});
            if (release_index >= rows.size()) {
                throw std::logic_error("packet simulation ran out of releases");
            }
            const TimePs next_release = rows[release_index].released_at_ps;
            if (next_release > calendar.nextSlotStartPs()) {
                calendar.rebaseIdle(next_release);
            }
        }
        const TimePs now_ps = calendar.nextSlotStartPs();
        enqueueReleased(now_ps);
        admitWaiting(now_ps);

        std::vector<RnicMaxMinFlow> active;
        active.reserve(active_by_class.size());
        for (const auto& item : active_by_class) {
            active.push_back({item.second, item.first.first, item.first.second});
        }
        const auto allocation = RnicMaxMinAllocator::allocate(
            active, source_capacities, destination_capacities);
        recordAllocation(active, allocation, options, ledger);
        std::vector<RnicPacketizedGrant> grants;
        grants.reserve(active.size());
        for (const RnicMaxMinFlow& flow : active) {
            grants.push_back(
                {flow.flow_id,
                 flow.source_node,
                 flow.destination_node,
                 allocation.at(flow.flow_id)});
        }
        calendar.beginEpoch(calendar.nextSlotIndex(), grants);
        ++ledger.allocation_epochs;
        const std::vector<RnicPacketizedReservation> reservations =
            calendar.reserveNextSlot();
        for (const RnicPacketizedReservation& reservation : reservations) {
            State& state = states.at(reservation.flowId());
            const RnicPacketExtent extent = packetization.packetize(
                state.flow.payload_bytes - state.payload_reserved);
            const RnicPacketizedTransmission packet =
                reservation.materializePacket(extent);
            state.payload_reserved += extent.payloadBytes();
            state.wire_bytes += extent.wireBytes();
            ++state.packet_count;
            if (state.payload_reserved != state.flow.payload_bytes) {
                continue;
            }
            state.completion_time_ps = packet.destinationSerializationEndPs();
            active_by_class.erase({state.flow.source, state.flow.destination});
            ++completed;
        }
    }

    std::vector<CompletionRow> output;
    output.reserve(rows.size());
    for (const auto& item : states) {
        const State& state = item.second;
        if (!state.completion_time_ps.has_value()) {
            throw std::logic_error("packet flow has no completion");
        }
        output.push_back(
            {"rnic-nn",
             state.flow,
             state.admitted_at_ps,
             *state.completion_time_ps,
             state.packet_count,
             state.wire_bytes});
        ledger.payload_bytes += state.flow.payload_bytes;
        ledger.packet_count += state.packet_count;
        ledger.wire_bytes += state.wire_bytes;
    }
    ledger.flow_count = output.size();
    return output;
}

std::vector<CompletionRow> simulateFluid(
        const std::vector<FlowRow>& rows,
        const Options& options,
        SimulationLedger& ledger) {
    RnicFluidManifold manifold(
        sourceCapacities(rows, options.source_capacity_bps),
        destinationCapacities(rows, options.destination_capacity_bps),
        options.propagation_ps);
    std::map<ClassKey, std::deque<FlowRow>> waiting;
    std::map<ClassKey, FlowId> active_by_class;
    std::map<FlowId, FlowRow> admitted_flows;
    std::map<FlowId, TimePs> admitted_at;
    std::map<FlowId, TimePs> completed_at;
    std::size_t release_index = 0;

    const auto collectCompletions = [&](TimePs now_ps) {
        std::vector<ClassKey> completed_classes;
        for (const auto& item : active_by_class) {
            const RnicFluidFlowSnapshot snapshot = manifold.flow(item.second);
            if (snapshot.active()) {
                continue;
            }
            if (!snapshot.delivery_completion_time_ps.has_value()) {
                throw std::logic_error("fluid completion has no delivery timestamp");
            }
            completed_at.emplace(item.second, *snapshot.delivery_completion_time_ps);
            completed_classes.push_back(item.first);
        }
        for (const ClassKey& key : completed_classes) {
            active_by_class.erase(key);
        }
        static_cast<void>(now_ps);
    };
    const auto enqueueReleased = [&](TimePs now_ps) {
        while (release_index < rows.size()
               && rows[release_index].released_at_ps <= now_ps) {
            const FlowRow& flow = rows[release_index++];
            waiting[{flow.source, flow.destination}].push_back(flow);
        }
    };
    const auto admitWaiting = [&](TimePs now_ps) {
        for (auto& item : waiting) {
            if (item.second.empty() || active_by_class.count(item.first) != 0) {
                continue;
            }
            const FlowRow flow = item.second.front();
            item.second.pop_front();
            manifold.addFlow(
                {flow.numeric_id, flow.source, flow.destination, flow.payload_bytes},
                now_ps);
            active_by_class.emplace(item.first, flow.numeric_id);
            admitted_flows.emplace(flow.numeric_id, flow);
            admitted_at.emplace(flow.numeric_id, now_ps);
        }
    };
    const auto recordRates = [&]() {
        std::vector<RnicMaxMinFlow> active;
        RnicMaxMinAllocator::AllocationMap allocation;
        for (const auto& item : active_by_class) {
            const RnicFluidFlowSnapshot snapshot = manifold.flow(item.second);
            active.push_back({item.second, item.first.first, item.first.second});
            allocation.emplace(item.second, snapshot.rate_bps);
        }
        recordAllocation(active, allocation, options, ledger);
        ++ledger.allocation_epochs;
    };

    while (completed_at.size() < rows.size()) {
        const std::optional<TimePs> next_completion =
            manifold.nextServiceCompletionTime();
        const std::optional<TimePs> next_release =
            release_index < rows.size()
                ? std::optional<TimePs>(rows[release_index].released_at_ps)
                : std::nullopt;
        if (!next_completion.has_value() && !next_release.has_value()) {
            throw std::logic_error("fluid simulation has no next event");
        }
        TimePs next_event = next_completion.value_or(*next_release);
        if (next_release.has_value()) {
            next_event = std::min(next_event, *next_release);
        }
        manifold.advanceTo(next_event);
        collectCompletions(next_event);
        enqueueReleased(next_event);
        admitWaiting(next_event);
        recordRates();
    }

    std::vector<CompletionRow> output;
    output.reserve(rows.size());
    for (const auto& item : admitted_flows) {
        const FlowRow& flow = item.second;
        output.push_back(
            {"rnic-nn-fluid",
             flow,
             admitted_at.at(item.first),
             completed_at.at(item.first),
             0,
             0});
        ledger.payload_bytes += flow.payload_bytes;
    }
    ledger.flow_count = output.size();
    return output;
}

void writeCompletions(const std::string& path, std::vector<CompletionRow> rows) {
    std::sort(rows.begin(), rows.end(), [](const CompletionRow& lhs, const CompletionRow& rhs) {
        return std::tie(lhs.transport, lhs.flow.numeric_id)
               < std::tie(rhs.transport, rhs.flow.numeric_id);
    });
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write completion CSV " + path);
    }
    output << "transport,numeric_id,flow_id,wave,source,destination,payload_bytes,"
              "released_at_ps,admitted_at_ps,completion_time_ps,fct_ps,class_wait_ps,"
              "packet_count,wire_bytes\n";
    for (const CompletionRow& row : rows) {
        if (row.completion_time_ps < row.flow.released_at_ps
            || row.admitted_at_ps < row.flow.released_at_ps) {
            throw std::logic_error("completion or admission precedes release");
        }
        output << row.transport << ',' << row.flow.numeric_id << ',' << row.flow.flow_id
               << ',' << row.flow.wave << ',' << row.flow.source << ','
               << row.flow.destination << ',' << row.flow.payload_bytes << ','
               << row.flow.released_at_ps << ',' << row.admitted_at_ps << ','
               << row.completion_time_ps << ','
               << row.completion_time_ps - row.flow.released_at_ps << ','
               << row.admitted_at_ps - row.flow.released_at_ps << ','
               << row.packet_count << ',' << row.wire_bytes << '\n';
    }
}

void writeLedger(std::ostream& output, const SimulationLedger& ledger, int indent) {
    const std::string spaces(static_cast<std::size_t>(indent), ' ');
    output << spaces << "\"flow_count\": " << ledger.flow_count << ",\n"
           << spaces << "\"payload_bytes\": " << ledger.payload_bytes << ",\n"
           << spaces << "\"packet_count\": " << ledger.packet_count << ",\n"
           << spaces << "\"wire_bytes\": " << ledger.wire_bytes << ",\n"
           << spaces << "\"allocation_epochs\": " << ledger.allocation_epochs << ",\n"
           << spaces << "\"max_active_classes\": " << ledger.max_active_classes << ",\n"
           << spaces << "\"max_source_allocated_bps\": "
           << ledger.max_source_allocated_bps << ",\n"
           << spaces << "\"max_destination_allocated_bps\": "
           << ledger.max_destination_allocated_bps << ",\n"
           << spaces << "\"active_pair_limit_violations\": "
           << ledger.active_pair_limit_violations << '\n';
}

void writeManifest(
        const std::string& path,
        const Options& options,
        const SimulationLedger& packet,
        const SimulationLedger& fluid) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write manifest JSON " + path);
    }
    output << "{\n"
           << "  \"schema\": \"simllm-traf72-rnic-adapter-manifest-v1\",\n"
           << "  \"htsim_source_commit\": \"" << HTSIM_SOURCE_COMMIT << "\",\n"
           << "  \"mapping\": \"one-active-transfer-per-ordered-pair-class\",\n"
           << "  \"source_capacity_bps\": " << options.source_capacity_bps << ",\n"
           << "  \"destination_capacity_bps\": "
           << options.destination_capacity_bps << ",\n"
           << "  \"max_wire_bytes\": " << options.max_wire_bytes << ",\n"
           << "  \"header_bytes\": " << options.header_bytes << ",\n"
           << "  \"propagation_ps\": " << options.propagation_ps << ",\n"
           << "  \"node_count\": " << options.node_count << ",\n"
           << "  \"rnic-nn\": {\n";
    writeLedger(output, packet, 4);
    output << "  },\n"
           << "  \"rnic-nn-fluid\": {\n";
    writeLedger(output, fluid, 4);
    output << "  },\n"
           << "  \"ack_events\": 0,\n"
           << "  \"reverse_control_bytes\": 0,\n"
           << "  \"non_data_events\": 0\n"
           << "}\n";
}

int run(const Options& options) {
    const std::vector<FlowRow> rows = readSchedule(options.schedule_csv);
    for (const FlowRow& row : rows) {
        if (row.source >= options.node_count || row.destination >= options.node_count) {
            throw std::runtime_error("schedule endpoint exceeds configured node count");
        }
    }
    SimulationLedger packet_ledger;
    SimulationLedger fluid_ledger;
    std::vector<CompletionRow> completions =
        simulatePacket(rows, options, packet_ledger);
    std::vector<CompletionRow> fluid = simulateFluid(rows, options, fluid_ledger);
    completions.insert(completions.end(), fluid.begin(), fluid.end());
    writeCompletions(options.completion_csv, std::move(completions));
    writeManifest(options.manifest_json, options, packet_ledger, fluid_ledger);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.provenance) {
            std::cout << "schema=simllm-traf72-rnic-adapter-provenance-v1\n"
                      << "htsim_source_commit=" << HTSIM_SOURCE_COMMIT << '\n'
                      << "packet_primitives=RnicMaxMinAllocator+RnicPacketizedSlotCalendar\n"
                      << "fluid_primitive=RnicFluidManifold\n"
                      << "mapping=one-active-transfer-per-ordered-pair-class\n";
            return 0;
        }
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "TRAF-72 adapter error: " << error.what() << '\n';
        return 2;
    }
}
