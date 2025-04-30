#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "common.h"
#include "util.hpp"

class Timer {
private:
    std::chrono::_V2::system_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::_V2::system_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    long long duration_us = 0;
    long long all_duration_us = 0;

public:
    Timer() {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void end() {
        end_time = std::chrono::high_resolution_clock::now();
        duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        all_duration_us += duration_us;
    }

    long long get_duration_us() {
        return duration_us;
    }

    long long get_all_duration_us() {
        return all_duration_us;
    }
};

class Logger {
private:
    std::string log_file;
    std::string get_current_time() {
        time_t now = time(nullptr);
        char buf[20];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
        return std::string(buf);
    }

public:
    Logger(std::string filename) {
        log_file = LOG_DIR + "/" + filename;
    }

    void log(int rank, std::string message) {
        std::ofstream out(log_file, std::ios::app);
        if (out.is_open()) {
            out << get_current_time() << " [Rank" << rank << "] " << message << "\n";
        }
    }
};

#endif
