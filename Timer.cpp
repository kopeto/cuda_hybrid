#include "Timer.hpp"

#include <string>
#include <chrono>
#include <cstdio>

Timer::Timer()
{
	m_StartTimePoint = std::chrono::high_resolution_clock::now();
}

Timer::Timer(uint64_t *result)
{
	m_StartTimePoint = std::chrono::high_resolution_clock::now();
	_duration = result;
}

Timer::~Timer()
{
	Stop();
}

void Timer::Stop()
{
	auto duration = duration_now();
	std::string scale = "us";
	double val = duration;
	if (duration > 1000.0)
	{
		val *= 0.001;
		scale = "ms";
		if (val > 1000.0)
		{
			val *= 0.001;
			scale = "sec";
		}
	}
	// double ms = duration * 0.001;
	// double sec = ms * 0.001;
	if (scale == "us")
		// printf(ANSI_COLOR_GREEN "(%d%s)\n" ANSI_COLOR_RESET, (uint32_t)val, scale.c_str());
		printf("(%d%s)\n" , (uint32_t)val, scale.c_str());
	else
		// printf(ANSI_COLOR_GREEN "(%.2lf%s)\n" ANSI_COLOR_RESET, val, scale.c_str());
		printf("(%.2lf%s)\n" , val, scale.c_str());
}

double Timer::duration_now()
{
	auto endTimePoint = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_StartTimePoint).time_since_epoch().count();
	auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endTimePoint).time_since_epoch().count();
	auto duration = end - start;
	if (_duration)
		*_duration = duration;
	return (double)duration / 1000.0;
}

void Timer::BENCHMARK_RATE(uint64_t score)
{
	auto duration = duration_now();
	//double ms = duration * 0.001;
	//double s = ms * 0.001;
	double rate = (double)score / duration;
	char scale = 'M';
	if (rate < 1.0)
	{
		rate *= 1000.0;
		scale = 'K';
		if (rate < 1.0)
		{
			rate *= 1000.0;
			scale = ' ';
		}
	}
	printf(ANSI_COLOR_CYAN "Proccessing rate: %.2f %cnode/sec \n" ANSI_COLOR_RESET, rate, scale);
}
