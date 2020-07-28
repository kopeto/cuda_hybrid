#ifndef _TIMER_HPP
#define _TIMER_HPP

#include <cstdio>
#include <chrono>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define PRINT_DURATION(duration,ms) \
	printf(ANSI_COLOR_GREEN "%ldus (%.3lfms)\n" ANSI_COLOR_RESET,duration,ms)

#define PRINT_MSEC(ms) \
	printf(ANSI_COLOR_GREEN "(%.3lfms)\n" ANSI_COLOR_RESET,ms)

class Timer{

public:
	Timer();
	Timer(uint64_t* result);
	Timer(uint64_t *result, bool p);
	~Timer();

	void Stop();
	double duration_now();
	void BENCHMARK_RATE(uint64_t);

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
	uint64_t *_duration = nullptr;
	bool print_results = true;
};

#endif 
//_TIMER_HPP