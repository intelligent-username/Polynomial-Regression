// include/util/CFile.h
// Tiny RAII FILE* wrapper (header-only for now)

#ifndef CFILE_H
#define CFILE_H

#include <string>
#include <cstdio>
#include <stdexcept>
#include <cerrno>
#include <cstring>

class CFile {
public:
    CFile() = default;
    explicit CFile(const std::string& path, const char* mode = "r") { open(path, mode); }
    ~CFile() { close(); }

    CFile(const CFile&) = delete;
    CFile& operator=(const CFile&) = delete;

    CFile(CFile&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    CFile& operator=(CFile&& other) noexcept {
        if (this != &other) {
            close();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void open(const std::string& path, const char* mode = "r") {
        close();
        handle_ = std::fopen(path.c_str(), mode);
        if (!handle_) {
            throw std::runtime_error("Failed to open file '" + path + "': " + std::strerror(errno));
        }
        path_ = path;
    }

    bool isOpen() const { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }

    std::string readAll() {
        if (!handle_) throw std::runtime_error("File not open");
        std::fseek(handle_, 0, SEEK_END);
        long size = std::ftell(handle_);
        if (size < 0) throw std::runtime_error("ftell failed");
        std::rewind(handle_);
        std::string buffer(static_cast<size_t>(size), '\0');
        size_t read = std::fread(buffer.data(), 1, static_cast<size_t>(size), handle_);
        buffer.resize(read);
        return buffer;
    }

    void write(const std::string& data) {
        if (!handle_) throw std::runtime_error("File not open");
        size_t written = std::fwrite(data.data(), 1, data.size(), handle_);
        if (written != data.size()) {
            throw std::runtime_error("Short write to file: " + path_);
        }
    }

    void flush() noexcept {
        if (handle_) std::fflush(handle_);
    }

    void close() {
        if (handle_) {
            std::fclose(handle_);
            handle_ = nullptr;
        }
    }

private:
    FILE* handle_ = nullptr;
    std::string path_;
};

#endif // CFILE_H
