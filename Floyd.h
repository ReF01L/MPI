class Floyd {
private:
    static int Min(int A, int B);

    void ProcessTermination(const int *, const int *) const;

    static void DummyDataInitialization(int *, int);

    void DataDistribution(int *, int *, int, int) const;

    void ResultCollection(int *, int *, int, int) const;

    void RowDistribution(int *, int, int, int *) const;

    void ParallelFloyd(int *, int, int);

    void ProcessInitialization(int *&, int *&, int &, int &) const;

    int ProcRank;
    int ProcNum;

public:
    void run(int argc, char **argv);
};
