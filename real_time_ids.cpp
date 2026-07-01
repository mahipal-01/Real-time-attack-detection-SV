#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <csignal>
#include <pcap.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/wait.h>
#include <sys/types.h>
#endif

#include <unordered_map>

#include "feature_extraction.h"
#include "onnx_model.h"
#include "kalman.h"
#include "timing_reconstruct.h"

constexpr uint16_t SV_ETHERTYPE = 0x88BA;
constexpr uint16_t VLAN_ETHERTYPE = 0x8100;
constexpr int WARMUP = 80;

constexpr const char* CLASS_NAMES[4] = {
    "normal", "manipulation", "timing/protocol", "traffic"
};

pcap_t* global_pcap_handle = nullptr;

void signal_handler(int) {
    if (global_pcap_handle) {
        std::cout << "\n[Info] Stop signal received.\n";
        pcap_breakloop(global_pcap_handle);
        global_pcap_handle = nullptr;
    }
}

#ifdef _WIN32
enum Color {
    COLOR_RESET = 7, COLOR_RED = 12, COLOR_GREEN = 10,
    COLOR_YELLOW = 14, COLOR_CYAN = 11, COLOR_MAGENTA = 13, COLOR_WHITE = 15,
};

void set_console_color(Color c) {
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(h, c);
}
#else
void set_console_color(int) {}
#endif

inline float read_float_be(const uint8_t* p) {
    uint32_t raw = (static_cast<uint32_t>(p[0]) << 24) |
                   (static_cast<uint32_t>(p[1]) << 16) |
                   (static_cast<uint32_t>(p[2]) << 8) |
                   static_cast<uint32_t>(p[3]);
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

inline uint32_t read_u32_be(const uint8_t* p, size_t len) {
    uint32_t value = 0;
    size_t parse_len = std::min(len, static_cast<size_t>(4));
    for (size_t i = 0; i < parse_len; ++i) value = (value << 8) | p[i];
    return value;
}

inline bool read_ber_length(const uint8_t* data, size_t max_len, size_t& offset, size_t& length) {
    if (offset >= max_len) return false;
    uint8_t first = data[offset++];
    if (first < 0x80) { length = first; return true; }
    size_t count = first & 0x7F;
    if (count == 0 || offset + count > max_len || count > 4) return false;
    length = 0;
    for (size_t i = 0; i < count; ++i) length = (length << 8) | data[offset++];
    return true;
}

inline double decode_refrTm_to_seconds(const uint8_t* refrTm) {
    uint32_t seconds = (static_cast<uint32_t>(refrTm[0]) << 24) |
                       (static_cast<uint32_t>(refrTm[1]) << 16) |
                       (static_cast<uint32_t>(refrTm[2]) << 8) |
                       static_cast<uint32_t>(refrTm[3]);
    uint32_t fraction = (static_cast<uint32_t>(refrTm[4]) << 16) |
                        (static_cast<uint32_t>(refrTm[5]) << 8) |
                        static_cast<uint32_t>(refrTm[6]);
    return static_cast<double>(seconds) + (static_cast<double>(fraction) / 16777216.0);
}

void fast_parse_sv(const uint8_t* pkt, size_t len, double cap_time, std::vector<ParsedRecord>& records) {
    if (len < 22) return;

    ParsedRecord base_record;
    base_record.capture_time_sec = cap_time;
    std::memcpy(base_record.dst_mac, pkt, 6);
    std::memcpy(base_record.src_mac, pkt + 6, 6);

    size_t offset = 12;
    uint16_t ethertype = (pkt[offset] << 8) | pkt[offset + 1];
    offset += 2;

    if (ethertype == VLAN_ETHERTYPE) {
        if (len < 26) return;
        base_record.has_vlan = true;
        uint16_t tci = (pkt[offset] << 8) | pkt[offset + 1];
        base_record.vlan_priority = (tci >> 13) & 0x07;
        base_record.vlan_id = tci & 0x0FFF;
        offset += 2;
        ethertype = (pkt[offset] << 8) | pkt[offset + 1];
        offset += 2;
    }

    if (ethertype != SV_ETHERTYPE || offset + 8 > len) return;

    base_record.app_id = (pkt[offset] << 8) | pkt[offset + 1];
    base_record.length = (pkt[offset + 2] << 8) | pkt[offset + 3];
    base_record.reserved1 = (pkt[offset + 4] << 8) | pkt[offset + 5];
    base_record.reserved2 = (pkt[offset + 6] << 8) | pkt[offset + 7];
    offset += 8;

    if (offset >= len || pkt[offset] != 0x60) return;
    offset++;

    size_t apdu_len;
    if (!read_ber_length(pkt, len, offset, apdu_len)) return;
    size_t apdu_end = std::min(len, offset + apdu_len);

    while (offset < apdu_end) {
        uint8_t tag = pkt[offset++];
        size_t tlv_len;
        if (!read_ber_length(pkt, apdu_end, offset, tlv_len)) break;
        size_t val_offset = offset;
        offset += tlv_len;

        if (tag == 0x80) {
            base_record.noAsdu = read_u32_be(pkt + val_offset, tlv_len);
        } else if (tag == 0xA2) {
            size_t seq_off = val_offset;
            while (seq_off < val_offset + tlv_len) {
                uint8_t seq_tag = pkt[seq_off++];
                size_t asdu_len;
                if (!read_ber_length(pkt, val_offset + tlv_len, seq_off, asdu_len)) break;
                if (seq_tag == 0x30) {
                    ParsedRecord asdu_rec = base_record;
                    size_t inner = seq_off;
                    while (inner < seq_off + asdu_len) {
                        uint8_t in_tag = pkt[inner++];
                        size_t in_len;
                        if (!read_ber_length(pkt, seq_off + asdu_len, inner, in_len)) break;
                        switch (in_tag) {
                            case 0x80: std::memcpy(asdu_rec.svID, pkt + inner, std::min(in_len, sizeof(asdu_rec.svID) - 1)); break;
                            case 0x81: std::memcpy(asdu_rec.DatSet, pkt + inner, std::min(in_len, sizeof(asdu_rec.DatSet) - 1)); break;
                            case 0x82: asdu_rec.smpCnt = read_u32_be(pkt + inner, in_len); break;
                            case 0x83: asdu_rec.confrev = read_u32_be(pkt + inner, in_len); break;
                            case 0x84: std::memcpy(asdu_rec.refrTm, pkt + inner, std::min(in_len, sizeof(asdu_rec.refrTm))); break;
                            case 0x85: asdu_rec.smpSynch = read_u32_be(pkt + inner, in_len); break;
                            case 0x86: asdu_rec.smpRate = read_u32_be(pkt + inner, in_len); break;
                            case 0x87: {
                                size_t count = std::min(static_cast<size_t>(8), in_len / 8);
                                for (size_t i = 0; i < count; ++i) {
                                    size_t ch_offset = inner + (i * 8);
                                    asdu_rec.channels[i] = read_float_be(pkt + ch_offset);
                                    asdu_rec.quality[i] = read_u32_be(pkt + ch_offset + 4, 4);
                                }
                                asdu_rec.valid = true;
                                break;
                            }
                        }
                        inner += in_len;
                    }
                    if (asdu_rec.valid) records.push_back(asdu_rec);
                }
                seq_off += asdu_len;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <interface> [--model s5|s51|rwkv|rwkv1] [--csv <path>] [--csv-interval <N>]\n"
                  << "       [--mitigation] [--kalman-thresh <F>] [--kalman-Qa <F>] [--kalman-R <F>]\n"
                  << "       [--flood-sustain <N>] [--flood-recover-rate <F>] [--flood-check <N>]\n";
        return 1;
    }

    std::string iface = argv[1];
    std::string model_name = "s51";
    std::string true_label = "N/A";
    std::string csv_path;
    int csv_interval = 1;
    bool mitigation_enabled = false;
    double kalman_thresh = 0.05;
    double kalman_q_amp = 1e-6;
    double kalman_q_dc = 1e-8;
    double kalman_r_meas = 1e-4;
    int flood_sustain = 150;
    double flood_recover_rate = 6.0;
    int flood_check_interval = 1000;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--true-label") == 0 && i + 1 < argc) {
            true_label = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_name = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--csv-interval") == 0 && i + 1 < argc) {
            csv_interval = std::stoi(argv[++i]);
            if (csv_interval < 1) csv_interval = 1;
        } else if (strcmp(argv[i], "--mitigation") == 0) {
            mitigation_enabled = true;
        } else if (strcmp(argv[i], "--kalman-thresh") == 0 && i + 1 < argc) {
            kalman_thresh = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--kalman-Qa") == 0 && i + 1 < argc) {
            kalman_q_amp = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--kalman-R") == 0 && i + 1 < argc) {
            kalman_r_meas = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--flood-sustain") == 0 && i + 1 < argc) {
            flood_sustain = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--flood-recover-rate") == 0 && i + 1 < argc) {
            flood_recover_rate = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--flood-check") == 0 && i + 1 < argc) {
            flood_check_interval = std::stoi(argv[++i]);
            if (flood_check_interval < 1) flood_check_interval = 1;
        }
    }

    std::string model_path;
    int num_features = 49;
    if (model_name == "s5") {
        model_path = "s5_streaming.onnx";
        num_features = 52;
    } else if (model_name == "rwkv") {
        model_path = "rwkv_streaming.onnx";
        num_features = 52;
    } else if (model_name == "rwkv1") {
        model_path = "rwkv1_streaming.onnx";
    } else {
        model_path = model_name + "_streaming.onnx";
    }
    OnnxModel model(model_path, 2, 64, num_features);
    KalmanManager kalman_mgr(kalman_thresh, kalman_q_amp, kalman_q_dc, kalman_r_meas);

    std::signal(SIGINT, signal_handler);

    char errbuf[PCAP_ERRBUF_SIZE];
    global_pcap_handle = pcap_create(iface.c_str(), errbuf);
    if (!global_pcap_handle) {
        std::cerr << "[Error] pcap_create: " << errbuf << "\n";
        return 1;
    }

    pcap_set_snaplen(global_pcap_handle, 65535);
    pcap_set_promisc(global_pcap_handle, 1);
    pcap_set_timeout(global_pcap_handle, 1);
    pcap_set_immediate_mode(global_pcap_handle, 1);
    pcap_set_buffer_size(global_pcap_handle, 1024 * 1024 * 10);

    int status = pcap_activate(global_pcap_handle);
    if (status != 0) {
        std::cerr << "[Error] pcap_activate: " << pcap_statustostr(status) << "\n";
        pcap_close(global_pcap_handle);
        return 1;
    }

    int linktype = pcap_datalink(global_pcap_handle);
    if (linktype != DLT_EN10MB) {
        std::cerr << "[Warning] Expected Ethernet (" << DLT_EN10MB << "), got " << linktype << "\n";
    }

    std::cout << "Listening on " << iface
              << " (model: " << model_name << ")\n"
              << "Mitigation: " << (mitigation_enabled ? "ON" : "OFF") << "\n";
    if (mitigation_enabled) {
        std::cout << "  Kalman: thresh=" << kalman_thresh
                  << " Qa=" << kalman_q_amp << " R=" << kalman_r_meas << "\n"
                  << "  Flood: sustain=" << flood_sustain
                  << " recover_rate=" << flood_recover_rate
                  << " check_interval=" << flood_check_interval << "\n";
    }
    std::cout << "Press Ctrl+C to stop.\n\n";

    uint64_t packet_count = 0;
    uint64_t class_counts[4] = {0};

    struct CallbackCtx {
        uint64_t* packet_count;
        uint64_t* class_counts;
        std::string* true_label;
        OnnxModel* model;
        std::ofstream* csv_out;
        int csv_interval;
        int num_features;
        double proc_sum = 0;
        uint64_t proc_count = 0;
        double proc_max = 0;
        int prev_pred_idx = -1;
        KalmanManager* kalman = nullptr;
        std::unordered_map<std::string, TimingReconstructor>* timing_recons = nullptr;
        bool mitigation_enabled = false;
        bool flood_skip_active = false;
        int traffic_count = 0;
        uint64_t flood_pkt_count = 0;
        double flood_check_time = 0.0;
        double flood_last_print_time = 0.0;
        double flood_rate = 0.0;
        int flood_sustain = 0;
        double flood_recover_rate = 0.0;
        int flood_check_interval = 0;
    };

    static auto packet_handler = +[](u_char* user, const struct pcap_pkthdr* header, const u_char* packet) {
        CallbackCtx* c = reinterpret_cast<CallbackCtx*>(user);

        using clock = std::chrono::high_resolution_clock;

        auto t_start = clock::now();

        double cap_time = static_cast<double>(header->ts.tv_sec) +
                          static_cast<double>(header->ts.tv_usec) / 1000000.0;

        // Latency measured from start of decode to prediction (capture-to-decode delay excluded)
        double steady_now = std::chrono::duration<double>(clock::now().time_since_epoch()).count();
        std::vector<ParsedRecord> records;
        fast_parse_sv(packet, header->caplen, cap_time, records);

        if (records.empty()) return;

        for (const auto& rec : records) {
            (*c->packet_count)++;

            // Flood skip mode: bypass feature extraction and ML entirely
            if (c->flood_skip_active) {
                c->flood_pkt_count++;
                if (c->flood_pkt_count % c->flood_check_interval == 0) {
                    double now = std::chrono::duration<double>(clock::now().time_since_epoch()).count();
                    double elapsed = now - c->flood_check_time;
                    if (elapsed > 0) {
                        double rate = static_cast<double>(c->flood_check_interval) / elapsed;
                        c->flood_rate = rate;
                        if (rate < c->flood_recover_rate || rate <= 5000.0) {
                            c->flood_skip_active = false;
                            c->traffic_count = 0;
                            std::cout << "[Info] Flood ended. Resuming full processing.\n";
                        } else if (now - c->flood_last_print_time >= 1.0) {
                            c->flood_last_print_time = now;
                            std::cout << "[FLOOD ACTIVE] rate:" << static_cast<int>(rate) << "/s\n";
                        }
                    }
                    c->flood_check_time = now;
                    c->flood_pkt_count = 0;
                }
                continue;
            }

            double channels_double[8];
            for (int i = 0; i < 8; ++i) channels_double[i] = rec.channels[i];

            auto t_feat_start = clock::now();
            float features[52];
            compute_features(
                channels_double, rec.smpCnt,
                decode_refrTm_to_seconds(rec.refrTm),
                rec.capture_time_sec, rec.smpSynch,
                rec.refrTm[7],
                mac_to_string(rec.src_mac),
                features, c->num_features
            );
            auto t_feat_end = clock::now();
            double feat_us = std::chrono::duration<double, std::micro>(t_feat_end - t_feat_start).count();

            auto t_ml_start = clock::now();
            auto [pred_idx, confidence] = c->model->predict(features, c->num_features);
            auto t_ml_end = clock::now();
            double ml_us = std::chrono::duration<double, std::micro>(t_ml_end - t_ml_start).count();

            double processing_us = std::chrono::duration<double, std::micro>(t_ml_end - t_start).count();

            // Timing reconstruction
            TimingResult timing_result;
            bool is_replay = false;
            if (c->mitigation_enabled && c->timing_recons) {
                std::string src_mac = mac_to_string(rec.src_mac);
                TimingReconstructor& tr = (*c->timing_recons)[src_mac];
                double refrTm_sec = decode_refrTm_to_seconds(rec.refrTm);
                timing_result = tr.process(
                    rec.smpCnt, refrTm_sec,
                    rec.refrTm[7], rec.smpSynch,
                    features[3], features[4],
                    features[1], features[2],
                    pred_idx == 2
                );
                is_replay = (timing_result.type == TimingAttackType::REPLAY);
            }

            // Kalman reconstruction check (manipulation OR replay timing)
            bool kalman_active = false;
            if (c->mitigation_enabled && c->kalman && *c->packet_count > 160) {
                double pu_channels[6] = {
                    channels_double[0] / I_BASE,
                    channels_double[1] / I_BASE,
                    channels_double[2] / I_BASE,
                    channels_double[4] / V_BASE,
                    channels_double[5] / V_BASE,
                    channels_double[6] / V_BASE,
                };
                kalman_active = c->kalman->process(pu_channels, pred_idx == 1 || is_replay);
            }

            // Traffic flood detection (when mitigation enabled)
            if (c->mitigation_enabled) {
                if (pred_idx == 3) {
                    c->traffic_count++;
                    if (c->traffic_count >= c->flood_sustain) {
                        c->flood_skip_active = true;
                        c->flood_pkt_count = 0;
                        c->flood_check_time = std::chrono::duration<double>(clock::now().time_since_epoch()).count();
                        c->flood_last_print_time = c->flood_check_time;
                        c->traffic_count = 0;
                        std::cout << "[Info] Flood detected. Entering skip mode.\n";
                    }
                } else {
                    c->traffic_count = 0;
                }
            }

            const char* prediction = CLASS_NAMES[pred_idx];

            c->class_counts[pred_idx]++;

            if (c->csv_out && c->csv_out->is_open() && *c->packet_count % c->csv_interval == 0) {
                static int csv_nf = 0;
                static bool csv_header_written = false;
                if (!csv_header_written) {
                    csv_nf = c->num_features;
                    *c->csv_out << "pkt,pred,conf";
                    for (int i = 0; i < csv_nf; ++i)
                        *c->csv_out << ",f" << i;
                    *c->csv_out << "\n";
                    csv_header_written = true;
                }
                *c->csv_out << *c->packet_count << ","
                            << pred_idx << ","
                            << std::fixed << std::setprecision(6) << confidence;
                for (int i = 0; i < csv_nf; ++i)
                    *c->csv_out << "," << std::fixed << std::setprecision(8) << features[i];
                *c->csv_out << "\n";
            }

            if (*c->packet_count > WARMUP) {
                c->proc_sum += processing_us;
                c->proc_count++;
                if (processing_us > c->proc_max) c->proc_max = processing_us;
            }

            if (c->prev_pred_idx != -1 && pred_idx != c->prev_pred_idx && *c->packet_count > WARMUP) {
                static const Color CLASS_COLORS[4] = {COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_MAGENTA};
                set_console_color(CLASS_COLORS[pred_idx]);
                std::cout << "[" << std::fixed << std::setprecision(3) << cap_time << "] ";
                if (timing_result.type == TimingAttackType::REPLAY)
                    std::cout << "[REPLAY] ";
                else if (timing_result.type == TimingAttackType::SEQ_MANIP)
                    std::cout << "[SEQ MANIP] ";
                else if (timing_result.type == TimingAttackType::TIME_SYNC)
                    std::cout << "[TIME SYNC] ";
                else if (timing_result.type == TimingAttackType::UNKNOWN)
                    std::cout << "[TIMING] ";
                if (kalman_active)
                    std::cout << "[RECONSTRUCTING] ";
                std::cout << "Pkt#" << *c->packet_count
                          << " Pred:" << CLASS_NAMES[c->prev_pred_idx]
                          << " -> " << CLASS_NAMES[pred_idx]
                          << " Conf:" << std::setprecision(3) << confidence;
                // Timing correction display
                if (timing_result.type == TimingAttackType::REPLAY) {
                    double refrTm_sec = decode_refrTm_to_seconds(rec.refrTm);
                    std::cout << " smpCnt:" << rec.smpCnt
                              << "->" << timing_result.recon_smpCnt
                              << " refrTm:" << std::setprecision(6) << refrTm_sec
                              << "->" << timing_result.recon_refrTm_seconds;
                } else if (timing_result.type == TimingAttackType::SEQ_MANIP) {
                    std::cout << " smpCnt:" << rec.smpCnt
                              << "->" << timing_result.recon_smpCnt;
                } else if (timing_result.type == TimingAttackType::TIME_SYNC) {
                    double refrTm_sec = decode_refrTm_to_seconds(rec.refrTm);
                    std::cout << " refrTm:" << std::setprecision(6) << refrTm_sec
                              << "->" << timing_result.recon_refrTm_seconds;
                    if (timing_result.quality_suspicious) std::cout << " quality:suspicious";
                    if (timing_result.synch_suspicious) std::cout << " synch:suspicious";
                }
                // Kalman correction display
                if (kalman_active && c->kalman) {
                    std::cout << " Ia:" << std::setprecision(3) << c->kalman->received_pu(0)
                              << "->" << std::setprecision(3) << c->kalman->predicted_pu(0)
                              << " Va:" << std::setprecision(3) << c->kalman->received_pu(3)
                              << "->" << std::setprecision(3) << c->kalman->predicted_pu(3);
                }
                std::cout << " Latency:" << processing_us << "us\n";
                set_console_color(COLOR_RESET);
            }
            c->prev_pred_idx = pred_idx;
        }
    };

    std::ofstream csv_file;
    if (!csv_path.empty()) {
        csv_file.open(csv_path);
        if (!csv_file.is_open())
            std::cerr << "[Warning] Failed to open " << csv_path << " for writing\n";
        else
            std::cout << "Dumping features to " << csv_path << " (every " << csv_interval << " packets)\n";
    }

    std::unordered_map<std::string, TimingReconstructor> timing_recons;
    CallbackCtx ctx{&packet_count, class_counts, &true_label, &model,
                    csv_file.is_open() ? &csv_file : nullptr, csv_interval, num_features};
    ctx.kalman = &kalman_mgr;
    ctx.timing_recons = &timing_recons;
    ctx.mitigation_enabled = mitigation_enabled;
    ctx.flood_sustain = flood_sustain;
    ctx.flood_recover_rate = flood_recover_rate;
    ctx.flood_check_interval = flood_check_interval;
    pcap_loop(global_pcap_handle, 0, packet_handler, reinterpret_cast<u_char*>(&ctx));

    if (global_pcap_handle) {
        pcap_close(global_pcap_handle);
    }

    uint64_t total_alerts = class_counts[1] + class_counts[2] + class_counts[3];
    std::cout << "\n[Summary] Packets processed: " << packet_count
              << " | Predictions: " << packet_count
              << " | Alerts: " << total_alerts;
    if (ctx.proc_count > 0) {
        double avg_proc = ctx.proc_sum / ctx.proc_count;
        std::cout << " | Avg Latency: " << std::fixed << std::setprecision(1) << avg_proc << "us"
                  << " | Max Latency: " << ctx.proc_max << "us";
    }
    std::cout << "\n";
    for (int i = 0; i < 4; ++i)
        std::cout << "  " << CLASS_NAMES[i] << ": " << class_counts[i] << "\n";

    return 0;
}
