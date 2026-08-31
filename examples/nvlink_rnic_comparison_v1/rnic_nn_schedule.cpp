// Exact-picosecond study adapter for the pinned htsim rnic-nn runtime.

#include "atlahs_flow_runtime.h"
#include "eventlist.h"
#include "rnic_packetized_manifold_runtime.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef HTSIM_SOURCE_COMMIT
#error "HTSIM_SOURCE_COMMIT is required"
#endif

namespace {

struct Options {
    std::string schedule_csv;
    std::string completion_csv;
    std::string manifest_json;
    std::uint64_t capacity_bps{0};
    std::uint64_t max_wire_bytes{0};
    std::uint64_t header_bytes{0};
    std::uint64_t propagation_ps{0};
    std::uint32_t node_count{0};
    bool provenance{false};
};

struct FlowRow {
    std::uint64_t numeric_id;
    std::string flow_id;
    std::uint32_t source;
    std::uint32_t destination;
    std::uint64_t payload_bytes;
    std::uint64_t released_at_ps;
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
        } else if (option == "--capacity-bps") {
            options.capacity_bps = parseUnsigned(option, value);
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
        || options.manifest_json.empty() || options.capacity_bps == 0
        || options.max_wire_bytes == 0 || options.node_count == 0) {
        throw std::invalid_argument("all schedule and physical options are required");
    }
    if (options.header_bytes >= options.max_wire_bytes) {
        throw std::invalid_argument("header bytes must be smaller than maximum wire bytes");
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
        || line != "numeric_id,flow_id,source,destination,payload_bytes,released_at_ps") {
        throw std::runtime_error("schedule CSV header is not the frozen schema");
    }
    std::vector<FlowRow> rows;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = splitCsv(line);
        if (fields.size() != 6) {
            throw std::runtime_error("schedule CSV row must have six fields");
        }
        const std::uint64_t source = parseUnsigned("source", fields[2]);
        const std::uint64_t destination = parseUnsigned("destination", fields[3]);
        if (source > std::numeric_limits<std::uint32_t>::max()
            || destination > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("schedule endpoint exceeds uint32_t");
        }
        rows.push_back(
            {parseUnsigned("numeric_id", fields[0]),
             fields[1],
             static_cast<std::uint32_t>(source),
             static_cast<std::uint32_t>(destination),
             parseUnsigned("payload_bytes", fields[4]),
             parseUnsigned("released_at_ps", fields[5])});
    }
    if (rows.empty()) {
        throw std::runtime_error("schedule CSV contains no flows");
    }
    std::vector<std::uint64_t> identities;
    identities.reserve(rows.size());
    for (const FlowRow& row : rows) {
        identities.push_back(row.numeric_id);
    }
    std::sort(identities.begin(), identities.end());
    if (std::adjacent_find(identities.begin(), identities.end()) != identities.end()) {
        throw std::runtime_error("schedule numeric flow IDs are not unique");
    }
    return rows;
}

class InjectionEvent final : public EventSource {
public:
    InjectionEvent(EventList& event_list,
                   simtime_picosec when,
                   std::function<void()> callback)
        : EventSource(event_list, "TRAF-71 exact release"),
          callback_(std::move(callback)) {
        EventList::sourceIsPending(*this, when);
    }

    void doNextEvent() override { callback_(); }
    bool isTraffic() override { return true; }

private:
    std::function<void()> callback_;
};

void writeCompletion(const std::string& path,
                     const std::vector<FlowRow>& rows,
                     const RnicPacketizedManifoldRuntime& runtime) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write completion CSV " + path);
    }
    output << "transport,numeric_id,flow_id,source,destination,payload_bytes,"
              "released_at_ps,completion_time_ps,fct_ps,packet_count,wire_bytes\n";
    for (const FlowRow& row : rows) {
        const RnicPacketizedFlowSnapshot snapshot = runtime.flow(row.numeric_id);
        if (!snapshot.delivery_completion_time_ps.has_value()) {
            throw std::runtime_error("rnic-nn flow did not complete");
        }
        const std::uint64_t completion = *snapshot.delivery_completion_time_ps;
        if (completion < row.released_at_ps) {
            throw std::runtime_error("rnic-nn completion precedes release");
        }
        output << "rnic-nn," << row.numeric_id << ',' << row.flow_id << ','
               << row.source << ',' << row.destination << ',' << row.payload_bytes
               << ',' << row.released_at_ps << ',' << completion << ','
               << completion - row.released_at_ps << ','
               << snapshot.total_packet_count << ',' << snapshot.total_wire_bytes
               << '\n';
    }
}

void writeManifest(const std::string& path,
                   const Options& options,
                   const std::vector<FlowRow>& rows,
                   const RnicPacketizedManifoldRuntime& runtime,
                   std::uint64_t completion_callbacks,
                   const std::map<AtlahsRuntimeEventKind, std::uint64_t>& event_counts,
                   std::uint64_t data_events,
                   std::uint64_t non_data_events) {
    std::uint64_t payload_bytes = 0;
    std::uint64_t packet_count = 0;
    std::uint64_t wire_bytes = 0;
    for (const FlowRow& row : rows) {
        const RnicPacketizedFlowSnapshot snapshot = runtime.flow(row.numeric_id);
        payload_bytes += row.payload_bytes;
        packet_count += snapshot.total_packet_count;
        wire_bytes += snapshot.total_wire_bytes;
    }
    const auto count = [&](AtlahsRuntimeEventKind kind) {
        const auto item = event_counts.find(kind);
        return item == event_counts.end() ? UINT64_C(0) : item->second;
    };
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write manifest JSON " + path);
    }
    output << "{\n"
           << "  \"schema\": \"simllm-traf71-rnic-adapter-manifest-v1\",\n"
           << "  \"transport\": \"rnic-nn\",\n"
           << "  \"runtime_class\": \"RnicPacketizedManifoldRuntime\",\n"
           << "  \"htsim_source_commit\": \"" << HTSIM_SOURCE_COMMIT << "\",\n"
           << "  \"capacity_bps\": " << options.capacity_bps << ",\n"
           << "  \"max_wire_bytes\": " << options.max_wire_bytes << ",\n"
           << "  \"header_bytes\": " << options.header_bytes << ",\n"
           << "  \"propagation_ps\": " << options.propagation_ps << ",\n"
           << "  \"node_count\": " << options.node_count << ",\n"
           << "  \"flow_count\": " << rows.size() << ",\n"
           << "  \"payload_bytes\": " << payload_bytes << ",\n"
           << "  \"packet_count\": " << packet_count << ",\n"
           << "  \"wire_bytes\": " << wire_bytes << ",\n"
           << "  \"completion_callbacks\": " << completion_callbacks << ",\n"
           << "  \"packet_tx_started\": "
           << count(AtlahsRuntimeEventKind::PacketTxStarted) << ",\n"
           << "  \"packet_tx_finished\": "
           << count(AtlahsRuntimeEventKind::PacketTxFinished) << ",\n"
           << "  \"packet_rx_arrived\": "
           << count(AtlahsRuntimeEventKind::PacketRxArrived) << ",\n"
           << "  \"packet_delivered\": "
           << count(AtlahsRuntimeEventKind::PacketDelivered) << ",\n"
           << "  \"data_events\": " << data_events << ",\n"
           << "  \"non_data_events\": " << non_data_events << ",\n"
           << "  \"ack_events\": 0,\n"
           << "  \"reverse_control_bytes\": 0,\n"
           << "  \"pending_physical_work\": "
           << (runtime.hasPendingPhysicalWork() ? "true" : "false") << "\n"
           << "}\n";
}

int run(const Options& options) {
    EventList& event_list = EventList::getTheEventList();
    EventList::setEndtime(std::numeric_limits<simtime_picosec>::max());
    const std::vector<FlowRow> rows = readSchedule(options.schedule_csv);
    for (const FlowRow& row : rows) {
        if (row.source >= options.node_count || row.destination >= options.node_count) {
            throw std::runtime_error("schedule endpoint exceeds configured node count");
        }
    }

    std::uint64_t completion_callbacks = 0;
    std::map<AtlahsRuntimeEventKind, std::uint64_t> event_counts;
    std::uint64_t data_events = 0;
    std::uint64_t non_data_events = 0;
    RnicPacketizedManifoldRuntime runtime(
        event_list,
        options.capacity_bps,
        RnicDataPacketizationConfig(options.max_wire_bytes, options.header_bytes),
        options.propagation_ps);
    runtime.setEventHandler([&](const AtlahsRuntimeEvent& event) {
        ++event_counts[event.kind];
        if (event.packet_kind == AtlahsRuntimePacketKind::Data) {
            ++data_events;
        } else {
            ++non_data_events;
        }
    });
    runtime.setup(options.node_count, [&](AtlahsFlowId) {
        ++completion_callbacks;
    });

    std::map<std::uint64_t, std::vector<FlowRow>> by_release;
    for (const FlowRow& row : rows) {
        by_release[row.released_at_ps].push_back(row);
    }
    std::vector<std::unique_ptr<InjectionEvent>> injections;
    injections.reserve(by_release.size());
    for (const auto& release : by_release) {
        injections.push_back(std::make_unique<InjectionEvent>(
            event_list,
            release.first,
            [&, group = release.second]() {
                for (const FlowRow& row : group) {
                    runtime.send(
                        {row.numeric_id,
                         row.source,
                         row.destination,
                         row.payload_bytes,
                         EventList::now(),
                         0,
                         0});
                }
            }));
    }
    while (EventList::doNextEvent()) {
    }
    if (completion_callbacks != rows.size() || runtime.hasPendingPhysicalWork()) {
        throw std::runtime_error("rnic-nn runtime did not reach physical quiescence");
    }
    writeCompletion(options.completion_csv, rows, runtime);
    writeManifest(
        options.manifest_json,
        options,
        rows,
        runtime,
        completion_callbacks,
        event_counts,
        data_events,
        non_data_events);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.provenance) {
            std::cout << "schema=simllm-traf71-rnic-adapter-provenance-v1\n"
                      << "htsim_source_commit=" << HTSIM_SOURCE_COMMIT << '\n'
                      << "runtime_class=RnicPacketizedManifoldRuntime\n"
                      << "transport=rnic-nn\n"
                      << "ack_pacing=absent\n";
            return 0;
        }
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "TRAF-71 adapter error: " << error.what() << '\n';
        return 2;
    }
}
