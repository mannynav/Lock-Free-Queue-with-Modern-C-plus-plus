
#include <iostream>
#include <ostream>
#include <vector>
#include <array>
#include <random>
#include <ranges>
#include "Timer.h"
#include <cmath>
#include <thread>
#include <mutex>
#include <algorithm>
#include <numbers>
#include <fstream>
#include <format>
#include <atomic>
#include <condition_variable>
#include "Timer.h"


// test settings
constexpr size_t WorkerCount = 4;
constexpr size_t BlockCount = 1000;
constexpr size_t BlockSize = 16'000;
constexpr size_t SubsetSize = BlockSize / WorkerCount;
constexpr size_t LightIterations = 2;
constexpr size_t HeavyIterations = 20;
constexpr double HeavyProbability = 0.15;

static_assert(BlockCount >= WorkerCount);
static_assert(BlockSize% WorkerCount == 0);

struct Task
{
	double value;
	bool heavy;

	unsigned int Process() const
	{
		const auto iterations = heavy ? HeavyIterations : LightIterations;
		auto intermediate = value;

		for (size_t i = 0; i < iterations; ++i)
		{
			const auto digits = static_cast<unsigned int>(std::abs(std::sin(std::cos(intermediate)) * 10'000'000)) % 100'000;
			intermediate = static_cast<double>(digits) / 10'000;
		}

		return static_cast<unsigned int>(std::exp(intermediate));

	}
};
struct BlockInfo
{
	std::array<float, WorkerCount> ThreadWorkTime;
	std::array<size_t, WorkerCount> HeaviesPerThread;
	float TotalBlockTime;
};

std::vector<std::array<Task, BlockSize>> UnstructuredData()
{
	std::minstd_rand rne; //starts off with default seed. sum should be the same
	std::bernoulli_distribution distribution{ HeavyProbability };
	std::uniform_real_distribution<> uniform_distributionn{ 0,std::numbers::pi };

	std::vector<std::array<Task, BlockSize>> blocks(BlockCount);

	for (auto& block : blocks)
	{
		std::ranges::generate(block, [&]() -> Task {
			return Task{ .value = uniform_distributionn(rne),.heavy = distribution(rne) };
			}
		);
	}

	return blocks;
}
std::vector<std::array<Task, BlockSize>> StructuredData()
{
	std::minstd_rand rne; //starts off with default seed. sum should be the same
	std::uniform_real_distribution<> uniform_distribution{ 0,std::numbers::pi };

	const int everyNth = int(1.0 / HeavyProbability);

	std::vector<std::array<Task, BlockSize>> blocks(BlockCount);

	for (auto& block : blocks)
	{
		std::ranges::generate(block, [&, i = 0]() mutable {
			const auto isHeavy = i++ % everyNth == 0;
			return Task{ .value = uniform_distribution(rne),.heavy = isHeavy };
			}
		);
	}
	return blocks;
}
std::vector<std::array<Task, BlockSize>> FrontLoadedData()
{
	auto blocks = StructuredData();

	for (auto& block : blocks) {
		std::ranges::partition(block, std::identity{}, &Task::heavy);
	}

	return blocks;

}


class Manager
{
public:

	Manager() : lock{ mtx } {}

	void SignalDone() //called from a worker
	{
		bool needsNotification = false;
		{
			std::lock_guard lk{ mtx };
			++doneCount;
			if (doneCount == WorkerCount)
			{
				needsNotification = true;
			}
		}
		if (needsNotification)
		{
			cv.notify_one(); //notify cv of the master that its time to wake up
		}
	}

	void WaitForAllDone()
	{
		cv.wait(lock, [this] {return doneCount == WorkerCount; });
		doneCount = 0;
	}

private:

	std::condition_variable cv;
	std::mutex mtx;
	std::unique_lock<std::mutex> lock;

	//shared memory
	int doneCount = 0;
};
class Worker
{
public:
	Worker(Manager* pMaster) : manager_ptr(pMaster), thread(&Worker::Run_, this) {}

	void SetJob(std::span<const Task> data)
	{
		{
			std::lock_guard lk{ mutex_ };
			input = data;
		}
		cv.notify_one();
	}

	void Kill()
	{
		{
			std::lock_guard lk{ mutex_ };
			IsDying_ = true;
		}
		cv.notify_one();
	}
	unsigned int GetResult() const
	{
		return ProcessAccumulation_;
	}

	float GetJobWorkTime() const
	{
		return JobTime_;
	}

	size_t GetNumHeavyItemsProcessed() const
	{
		return NumberHeavyItemsProcessed_;
	}

	~Worker()
	{
		Kill();
	}

private:

	void ProcessData_()
	{
		NumberHeavyItemsProcessed_ = 0;
		for (const auto& task : input)
		{
			ProcessAccumulation_ += task.Process();
			NumberHeavyItemsProcessed_ += task.heavy ? 1 : 0;
		}
	}

	void Run_()
	{
		std::unique_lock lk{ mutex_ };
		while (true)
		{
			Timer timer;
			cv.wait(lk, [this] {return !input.empty() || IsDying_; });

			//once we are awake
			if (IsDying_)
			{
				break;
			}

			timer.Mark();

			//we must have some work
			ProcessData_();

			JobTime_ = timer.Peek();

			input = {};
			manager_ptr->SignalDone();
		}
	}

	Manager* manager_ptr;
	std::jthread thread;
	std::condition_variable cv;
	std::mutex mutex_;

	//shared memory
	std::span<const Task> input;
	unsigned int ProcessAccumulation_ = 0;
	bool IsDying_ = false;
	float JobTime_ = -1.f;

	size_t NumberHeavyItemsProcessed_{};

};
int RunExperiment(bool stacked)
{

	const auto blocks = [=]
		{
			if (stacked)
			{
				return FrontLoadedData();
			}
			else
			{
				return StructuredData();
			}
		}();


	Timer TotalTimer;
	TotalTimer.Mark();

	Manager manager;

	std::vector<std::unique_ptr<Worker>> WorkerPtrs;

	for (size_t i = 0; i < WorkerCount; ++i)
	{
		WorkerPtrs.push_back(std::make_unique<Worker>(&manager));
	}

	std::vector<BlockInfo> timings;
	timings.reserve(BlockCount);

	Timer BlockTimer;

	for (auto block : blocks)
	{
		BlockTimer.Mark();
		for (size_t iSubset = 0; iSubset < WorkerCount; ++iSubset)
		{
			WorkerPtrs[iSubset]->SetJob(std::span{ &block[iSubset * SubsetSize], SubsetSize });
		}
		manager.WaitForAllDone();
		const auto BlockTime = BlockTimer.Peek();

		timings.push_back({});
		for (size_t i = 0; i < WorkerCount; i++)
		{
			timings.back().HeaviesPerThread[i] = WorkerPtrs[i]->GetNumHeavyItemsProcessed();
			timings.back().ThreadWorkTime[i] = WorkerPtrs[i]->GetJobWorkTime();
		}
		timings.back().TotalBlockTime = BlockTime;
	}

	const auto t = TotalTimer.Peek();

	std::cout << "Generating data took: " << t << "seconds" << std::endl;

	unsigned int finalResult = 0;

	for (const auto& w : WorkerPtrs)
	{
		finalResult += w->GetResult();
	}

	std::cout << "result is: " << finalResult << std::endl;

	//output csv of chunk timings
	// worktime, idletime, numberofheavies x workers + total time, total heavies

	std::ofstream csv("ResultsTime.csv", std::ios_base::trunc);

	for (size_t i = 0; i < WorkerCount; i++)
	{
		csv << std::format("work_{0:},idle{0:},heavy_{0:},", i);
	}

	csv << "chunktime, total_idle, total_heavy\n";

	for (const auto& chunk : timings)
	{
		float totalIdle = 0;
		size_t totalHeavy = 0;

		for (size_t i = 0; i < WorkerCount; i++)
		{
			const auto idle = chunk.TotalBlockTime - chunk.ThreadWorkTime[i];
			const auto heavy = chunk.HeaviesPerThread[i];
			csv << std::format("{},{},{},", chunk.ThreadWorkTime[i], idle, heavy);
			totalIdle += idle;
			totalHeavy += heavy;
		}

		csv << std::format("{},{},{}\n", chunk.TotalBlockTime, totalIdle, totalHeavy);

	}

	return 0;

}


class QueueManager
{
public:

	QueueManager() : lk{ mtx } {}

	void SignalDone() //called from a worker
	{
		bool needsNotification = false;
		{
			std::lock_guard lk{ mtx };
			++doneCount;
			if (doneCount == WorkerCount)
			{
				needsNotification = true;
			}
		}
		if (needsNotification)
		{
			cv.notify_one(); //notify cv of the master that its time to wake up
		}
	}

	void WaitForAllDone()
	{
		cv.wait(lk, [this] {return doneCount == WorkerCount; });
		doneCount = 0;
	}

	void SetChunk(std::span<const Task> chunk)
	{
		index = 0; //start at top of chunk
		currentChunk = chunk; //set the chunk
	}

	const Task* GetTask()
	{
		//std::lock_guard lock{ mutex_ };
		const auto i = index++; //return current index and increment. Do not need fetch_add(1)
		if (i >= BlockSize)
		{
			return nullptr;
		}

		return &currentChunk[i]; //returns pointer to the next task to be processed.
	}

private:

	std::condition_variable cv;
	std::mutex mtx;
	std::unique_lock<std::mutex> lk;
	std::span<const Task> currentChunk;

	//shared memory
	int doneCount = 0;
	//size_t index = 0;
	std::atomic<size_t> index = 0;
};
class QueuedWorker
{
public:

	QueuedWorker(QueueManager* manager_ptr) : manager_ptr(manager_ptr), thread(&QueuedWorker::Run_, this) {}

	void StartWork()
	{
		{
			std::lock_guard lk{ mtx };
			working = true;
		}
		cv.notify_one();
	}

	void Kill()
	{
		{
			std::lock_guard lk{ mtx };
			dying = true;
		}
		cv.notify_one();
	}
	unsigned int GetResult() const
	{
		return accumulation;
	}

	float GetJobWorkTime() const
	{
		return workTime;
	}

	size_t GetNumHeavyItemsProcessed() const
	{
		return numHeavyItemsProcessed;
	}

	~QueuedWorker()
	{
		Kill();
	}

private:

	void ProcessData_()
	{
		numHeavyItemsProcessed = 0;

		while (auto pTask = manager_ptr->GetTask()) //while we return a task that is not null
		{
			accumulation += pTask->Process();
			numHeavyItemsProcessed += pTask->heavy ? 1 : 0;
		}
	}
	void Run_()
	{
		std::unique_lock lk{ mtx };
		while (true)
		{
			Timer timer;
			cv.wait(lk, [this] {return working || dying; });

			//once we are awake
			if (dying)
			{
				break;
			}

			timer.Mark();

			//we must have some work
			ProcessData_();

			workTime = timer.Peek();

			working = false;

			manager_ptr->SignalDone();
		}
	}

	QueueManager* manager_ptr;
	std::jthread thread;
	std::condition_variable cv;
	std::mutex mtx;

	//shared memory
	unsigned int accumulation = 0;
	bool dying = false;
	float workTime = -1.f;
	size_t numHeavyItemsProcessed{};
	bool working = false;

};
int RunQueuedExperiment(bool stacked)
{

	const auto chunks = [=]
		{
			if (stacked)
			{
				return FrontLoadedData();
			}
			else
			{
				return StructuredData();
			}
		}();


	Timer totalTimer;
	totalTimer.Mark();

	QueueManager mctrl;

	std::vector<std::unique_ptr<QueuedWorker>> WorkerPtrs(WorkerCount);

	std::ranges::generate(WorkerPtrs, [pMctr1 = &mctrl] {return std::make_unique<QueuedWorker>(pMctr1); });

	std::vector<BlockInfo> timings;
	timings.reserve(BlockCount);

	Timer chunkTimer;

	for (auto chunk : chunks)
	{
		chunkTimer.Mark();
		mctrl.SetChunk(chunk);

		for (auto& pWorker : WorkerPtrs)
		{
			pWorker->StartWork();
		}
		mctrl.WaitForAllDone();

		const auto chunkTime = chunkTimer.Peek();

		timings.push_back({});
		for (size_t i = 0; i < WorkerCount; i++)
		{
			timings.back().HeaviesPerThread[i] = WorkerPtrs[i]->GetNumHeavyItemsProcessed();
			timings.back().ThreadWorkTime[i] = WorkerPtrs[i]->GetJobWorkTime();
		}
		timings.back().TotalBlockTime = chunkTime;

	}

	const auto t = totalTimer.Peek();

	std::cout << "Generating data took: " << t << "seconds" << std::endl;

	unsigned int finalResult = 0;

	for (const auto& w : WorkerPtrs)
	{
		finalResult += w->GetResult();
	}

	std::cout << "result is: " << finalResult << std::endl;


	std::ofstream csv("ResultsTimerQueued.csv", std::ios_base::trunc);

	for (size_t i = 0; i < WorkerCount; i++)
	{
		csv << std::format("work_{0:},idle{0:},heavy_{0:},", i);
	}

	csv << "chunktime, total_idle, total_heavy\n";

	for (const auto& chunk : timings)
	{
		float totalIdle = 0;
		size_t totalHeavy = 0;

		for (size_t i = 0; i < WorkerCount; i++)
		{
			const auto idle = chunk.TotalBlockTime - chunk.ThreadWorkTime[i];
			const auto heavy = chunk.HeaviesPerThread[i];
			csv << std::format("{},{},{},", chunk.ThreadWorkTime[i], idle, heavy);
			totalIdle += idle;
			totalHeavy += heavy;
		}

		csv << std::format("{},{},{}\n", chunk.TotalBlockTime, totalIdle, totalHeavy);

	}
	return 0;
}


int main(int argc, char** argv)
{
	using namespace std::string_literals;
	bool stacked = false;

	return RunExperiment(stacked);
}