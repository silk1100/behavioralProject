import threading


class MyThread(threading.Thread):
    def __init__(self, threadid, threadname, fnc):
        threading.Thread.__init__(self)
        self.threadID = threadid
        self.name = threadname
        self.fnc = fnc

    def run(self) -> None:
        print(f'Starting {self.name}')
        try:
            self.fnc()
        except:
            pass
        print(f'Finished {self.name}')


def counter():
    print(list(range(20)))


if __name__ == "__main__":
    thread_list = [MyThread(i, f"thread-{i}", counter) for i in range(10)]
    _ = [t.start() for t in thread_list]
    _ = [t.join() for t in thread_list]

    # th1.start()
    # th2.start()
    # th1.join()
    # th2.join()
    print("Finish main thread")
