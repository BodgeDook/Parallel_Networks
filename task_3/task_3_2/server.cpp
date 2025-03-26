#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>

template<typename ResultType>
class TaskServer {
public:
    TaskServer() : active(false), task_id_counter(0) {}

    void launch() {
        active = true;
        worker = std::thread(&TaskServer::handle_tasks, this);
    }

    void shutdown() {
        active = false;
        notifier.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    size_t submit_task(std::function<ResultType()> task) {
        std::lock_guard<std::mutex> guard(sync);
        size_t task_id = task_id_counter++;
        tasks.emplace(task_id, std::async(std::launch::deferred, task));
        notifier.notify_one();
        return task_id;
    }

    ResultType get_result(size_t task_id) {
        std::unique_lock<std::mutex> lock(sync);
        notifier.wait(lock, [this, task_id] { return completed_tasks.count(task_id) > 0; });
        return completed_tasks[task_id];
    }

private:
    void handle_tasks() {
        while (active || !tasks.empty()) {
            std::unique_lock<std::mutex> lock(sync);
            if (tasks.empty()) {
                notifier.wait(lock, [this] { return !tasks.empty() || !active; });
            }
            if (!tasks.empty()) {
                auto current_task = std::move(tasks.front());
                tasks.pop();
                lock.unlock();
                ResultType value = current_task.second.get();
                lock.lock();
                completed_tasks[current_task.first] = value;
                notifier.notify_all();
            }
        }
    }

    std::thread worker;
    bool active;
    std::queue<std::pair<size_t, std::future<ResultType>>> tasks;
    std::unordered_map<size_t, ResultType> completed_tasks;
    std::mutex sync;
    std::condition_variable notifier;
    size_t task_id_counter;
};

void sin_client(TaskServer<double>& server, int num_tasks, const std::string& output_file) {
    std::ofstream out(output_file);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<> range(-3.14, 3.14);

    int count = 0;
    while (count < num_tasks) {
        double x = range(rng);
        size_t task_id = server.submit_task([x] { return std::sin(x); });
        double answer = server.get_result(task_id);
        out << "sin(" << x << ") = " << answer << "\n";
        ++count;
    }
}

void sqrt_client(TaskServer<double>& server, int num_tasks, const std::string& output_file) {
    std::ofstream out(output_file);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<> range(0, 100);

    int count = 0;
    while (count < num_tasks) {
        double x = range(rng);
        size_t task_id = server.submit_task([x] { return std::sqrt(x); });
        double answer = server.get_result(task_id);
        out << "sqrt(" << x << ") = " << answer << "\n";
        ++count;
    }
}

void pow_client(TaskServer<double>& server, int num_tasks, const std::string& output_file) {
    std::ofstream out(output_file);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<> range(1, 10);

    int count = 0;
    while (count < num_tasks) {
        double base = range(rng);
        double exponent = range(rng);
        size_t task_id = server.submit_task([base, exponent] { return std::pow(base, exponent); });
        double answer = server.get_result(task_id);
        out << base << "^" << exponent << " = " << answer << "\n";
        ++count;
    }
}

void verify_results(const std::string& file_path, const std::string& operation) {
    std::ifstream input(file_path);
    std::string line;

    while (std::getline(input, line)) {
        std::istringstream parser(line);
        double input1, input2, computed;
        char separator;

        if (operation == "sin") {
            std::string label;
            parser >> label >> input1 >> separator >> computed;
            double expected = std::sin(input1);
            if (std::abs(computed - expected) > 1e-8) {
                std::cerr << "Mismatch in " << file_path << ": " << line << " | Expected: " << expected << "\n";
            }
        } 
        else if (operation == "sqrt") {
            std::string label;
            parser >> label >> input1 >> separator >> computed;
            double expected = std::sqrt(input1);
            if (std::abs(computed - expected) > 1e-8) {
                std::cerr << "Mismatch in " << file_path << ": " << line << " | Expected: " << expected << "\n";
            }
        } 
        else if (operation == "pow") {
            char caret;
            parser >> input1 >> caret >> input2 >> separator >> computed;
            double expected = std::pow(input1, input2);
            if (std::abs(computed - expected) > 1e-1) {
                std::cerr << "Mismatch in " << file_path << ": " << line << " | Expected: " << expected << "\n";
            }
        }
    }
}

int main() {

    TaskServer<double> task_server;
    task_server.launch();

    std::thread sin_thread(sin_client, std::ref(task_server), 2000, "sin_results.txt");
    std::thread sqrt_thread(sqrt_client, std::ref(task_server), 2000, "sqrt_results.txt");
    std::thread pow_thread(pow_client, std::ref(task_server), 2000, "pow_results.txt");

    sin_thread.join();
    sqrt_thread.join();
    pow_thread.join();

    task_server.shutdown();

    verify_results("sin_results.txt", "sin");
    verify_results("sqrt_results.txt", "sqrt");
    verify_results("pow_results.txt", "pow");

    std::cout << "Verification done.\n";

    return 0;
}