#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define SQR(x) ((x) * (x))

// Global state variables
int NowYear = 2025;
int NowMonth = 0;
float NowPrecip;
float NowTemp;
float NowHeight = 5.0;
int NowNumDeer = 2;
int NowNumPests = 10;

// Simulation parameters
const float GRAIN_GROWS_PER_MONTH = 13.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;
const float AVG_PRECIP_PER_MONTH = 7.0;
const float AMP_PRECIP_PER_MONTH = 6.0;
const float RANDOM_PRECIP = 1.0;
const float AVG_TEMP = 60.0;
const float AMP_TEMP = 20.0;
const float RANDOM_TEMP = 5.0;
const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;
const float ONE_DEER_EATS_PESTS = 1.0;
const float PESTS_EAT_GRAIN = 0.3;

// Random number generator
unsigned int seed = 0;
float Ranf(float low, float high) {
    float r = (float)rand_r(&seed) / (float)RAND_MAX;
    return low + r * (high - low);
}

// Barrier globals
omp_lock_t Lock;
volatile int NumInThreadTeam;
volatile int NumAtBarrier;
volatile int NumGone;

// Barrier functions
void InitBarrier(int n) {
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock(&Lock);
}

void WaitBarrier() {
    omp_set_lock(&Lock);
    {
        NumAtBarrier++;
        if (NumAtBarrier == NumInThreadTeam) {
            NumGone = 0;
            NumAtBarrier = 0;
            while (NumGone != NumInThreadTeam - 1);
            omp_unset_lock(&Lock);
            return;
        }
    }
    omp_unset_lock(&Lock);
    while (NumAtBarrier != 0);
    #pragma omp atomic
    NumGone++;
}

// Thread functions
void Deer() {
    while (NowYear < 2031) {
        int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)NowHeight;
        if (nextNumDeer < carryingCapacity)
            nextNumDeer++;
        else if (nextNumDeer > carryingCapacity)
            nextNumDeer--;
        if (nextNumDeer < 0)
            nextNumDeer = 0;

        WaitBarrier(); // DoneComputing
        NowNumDeer = nextNumDeer;
        WaitBarrier(); // DoneAssigning
        WaitBarrier(); // DonePrinting
    }
}

void Grain() {
    while (NowYear < 2031) {
        float tempFactor = exp(-SQR((NowTemp - MIDTEMP) / 10.));
        float precipFactor = exp(-SQR((NowPrecip - MIDPRECIP) / 10.));
        float nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        nextHeight -= (float)NowNumPests * PESTS_EAT_GRAIN;
        if (nextHeight < 0.)
            nextHeight = 0.;

        WaitBarrier(); // DoneComputing
        NowHeight = nextHeight;
        WaitBarrier(); // DoneAssigning
        WaitBarrier(); // DonePrinting
    }
}

void Watcher() {
    FILE *file = fopen("simulation.csv", "w");
    if (file == NULL) {
        printf("Error: Could not open simulation.csv for writing.\n");
        exit(1);
    }
    fprintf(file, "Month,Year,Temp(°C),Precip(cm),Deer,Grain(cm),Pests\n");

    while (NowYear < 2031) {
        WaitBarrier(); // DoneComputing
        WaitBarrier(); // DoneAssigning

        // Print the current state
        float tempC = (5. / 9.) * (NowTemp - 32.);
        float precipCm = NowPrecip * 2.54;
        float heightCm = NowHeight * 2.54;
        fprintf(file, "%d,%d,%.2f,%.2f,%d,%.2f,%d\n",
                NowMonth, NowYear, tempC, precipCm, NowNumDeer, heightCm, NowNumPests);

        // Increment month and year before updating Temp and Precip for the next iteration
        NowMonth++;
        if (NowMonth == 12) {
            NowMonth = 0;
            NowYear++;
        }

        // Update Temp and Precip for the next month
        float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);
        float temp = AVG_TEMP - AMP_TEMP * cos(ang);
        NowTemp = temp + Ranf(-RANDOM_TEMP, RANDOM_TEMP);
        if (NowTemp < 30.0) NowTemp = 30.0;
        if (NowTemp > 80.0) NowTemp = 80.0;
        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
        NowPrecip = precip + Ranf(-RANDOM_PRECIP, RANDOM_PRECIP);
        if (NowPrecip < 0.)
            NowPrecip = 0.;

        WaitBarrier(); // DonePrinting
    }
    fclose(file);
}

void MyAgent() {
    while (NowYear < 2031) {
        int nextNumPests = NowNumPests;
        float pestGrowthFactor = exp(-SQR((NowTemp - 50.0) / 15.));
        nextNumPests += (int)(pestGrowthFactor * 7.0);
        nextNumPests -= NowNumDeer * ONE_DEER_EATS_PESTS;
        if (nextNumPests > 20 * (int)NowHeight)
            nextNumPests = 20 * (int)NowHeight;
        if (nextNumPests < 0)
            nextNumPests = 0;

        WaitBarrier(); // DoneComputing
        NowNumPests = nextNumPests;
        WaitBarrier(); // DoneAssigning
        WaitBarrier(); // DonePrinting
    }
}

int main() {
    // Initialize temperature and precipitation for Month 0
    float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);
    float temp = AVG_TEMP - AMP_TEMP * cos(ang);
    NowTemp = temp + Ranf(-RANDOM_TEMP, RANDOM_TEMP);
    if (NowTemp < 30.0) NowTemp = 30.0;
    if (NowTemp > 80.0) NowTemp = 80.0;
    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
    NowPrecip = precip + Ranf(-RANDOM_PRECIP, RANDOM_PRECIP);
    if (NowPrecip < 0.)
        NowPrecip = 0.;

    srand(time(NULL));
    omp_set_num_threads(4);
    InitBarrier(4);

    #pragma omp parallel sections
    {
        #pragma omp section
        { Deer(); }
        #pragma omp section
        { Grain(); }
        #pragma omp section
        { Watcher(); }
        #pragma omp section
        { MyAgent(); }
    }

    return 0;
}