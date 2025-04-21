#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#ifndef F_PI
#define F_PI        (float)M_PI
#endif

#ifndef DEBUG
#define DEBUG       false
#endif

#ifndef NUMT
#define NUMT            2
#endif

#ifndef NUMTRIALS
#define NUMTRIALS   100
#endif

#ifndef NUMTRIES
#define NUMTRIES    30
#endif

// Constants for the simulation ranges
const float GMIN =  10.0;  // ground distance in meters
const float GMAX =  20.0;  // ground distance in meters
const float HMIN =  20.0;  // cliff height in meters
const float HMAX =  30.0;  // cliff height in meters
const float DMIN =  10.0;  // distance to castle in meters
const float DMAX =  20.0;  // distance to castle in meters
const float VMIN =  20.0;  // initial cannonball velocity in meters/sec
const float VMAX =  30.0;  // initial cannonball velocity in meters/sec
const float THMIN =  70.0; // cannonball launch angle in degrees
const float THMAX =  80.0; // cannonball launch angle in degrees

const float GRAVITY =   -9.8;  // acceleration due to gravity in meters/sec^2
const float TOL = 5.0;         // tolerance in cannonball hitting the castle in meters

// Helper function to generate random float between low and high
float Ranf(float low, float high) {
    float r = (float)rand();  // 0 - RAND_MAX
    float t = r / (float)RAND_MAX;  // 0.0 - 1.0
    return low + t * (high - low);
}

// Helper function to generate random int between low and high
int Ranf(int low, int high) {
    return (int)Ranf((float)low, (float)high);
}

// Function to seed the random number generator based on current time
void TimeOfDaySeed() {
    time_t now;
    time(&now);
    struct tm n = *localtime(&now);
    struct tm jan01 = *localtime(&now);
    jan01.tm_mon = 0; jan01.tm_mday = 1;
    jan01.tm_hour = 0; jan01.tm_min = 0; jan01.tm_sec = 0;
    double seconds = difftime(now, mktime(&jan01));
    unsigned int seed = (unsigned int)(1000. * seconds);
    srand(seed);
}

// Degrees to radians
inline float Radians(float degrees) {
    return (F_PI / 180.f) * degrees;
}

int main(int argc, char* argv[]) {
#ifndef _OPENMP
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

    TimeOfDaySeed();  // Seed the random number generator
    omp_set_num_threads(NUMT);  // Set the number of threads to use in parallelizing the for-loop
    
    // Arrays to store random values for trials
    float *vs = new float[NUMTRIALS];
    float *ths = new float[NUMTRIALS];
    float *gs = new float[NUMTRIALS];
    float *hs = new float[NUMTRIALS];
    float *ds = new float[NUMTRIALS];

    // Fill the random-value arrays
    for (int n = 0; n < NUMTRIALS; n++) {
        vs[n] = Ranf(VMIN, VMAX);
        ths[n] = Ranf(THMIN, THMAX);
        gs[n] = Ranf(GMIN, GMAX);
        hs[n] = Ranf(HMIN, HMAX);
        ds[n] = Ranf(DMIN, DMAX);
    }

    double maxPerformance = 0.;  // Record max performance
    int numHits = 0;            // Record number of hits

    // Loop to discover the maximum performance
    for (int tries = 0; tries < NUMTRIES; tries++) {
        double time0 = omp_get_wtime();
        numHits = 0;

        #pragma omp parallel for reduction(+:numHits)
        for (int n = 0; n < NUMTRIALS; n++) {
            // Randomize the values for each trial
            float v = vs[n];
            float th = Radians(ths[n]);
            float vx = v * cos(th);
            float vy = v * sin(th);
            float g = gs[n];
            float h = hs[n];
            float d = ds[n];

            // Time to hit the ground
            float t1 = (-vy / (0.5 * GRAVITY));
            float x = vx * t1;
            if (x <= g) continue;  // Ball doesn't reach the cliff

            // Time to reach the cliff face
            float t2 = g / vx;
            float y = vy * t2 + 0.5 * GRAVITY * t2 * t2;
            if (y <= h) continue;  // Ball doesn't hit the vertical cliff face

            // Solve the quadratic equation for the time to reach the upper deck
            float A = 0.5 * GRAVITY;
            float B = vy;
            float C = -h;
            float disc = B * B - 4.f * A * C;  // Discriminant

            if (disc < 0.) continue;  // Ball doesn't reach the upper deck

            // Get the intersection time
            float sqrtdisc = sqrtf(disc);
            float t_up = (-B + sqrtdisc) / (2.f * A);
            float t_down = (-B - sqrtdisc) / (2.f * A);
            float tmax = fmax(t_up, t_down);

            // Calculate horizontal distance to the castle
            float upperDist = vx * tmax - g;

            // Check if the ball hits the castle
            if (fabs(upperDist - d) <= TOL) {
                numHits++;
            }
        }

        double time1 = omp_get_wtime();
        double timeDiff = time1 - time0;
        double megaTrialsPerSecond = 0.0;
        if (timeDiff > 1e-6) {  // Avoid division by zero or near-zero
            megaTrialsPerSecond = (double)NUMTRIALS / timeDiff / 1000000.;
        }
        if (megaTrialsPerSecond > maxPerformance) {
            maxPerformance = megaTrialsPerSecond;
        }
    }

    // Calculate probability of hitting the castle
    float probability = (float)numHits / (float)NUMTRIALS;

    // Print out results in CSV format
    #ifdef CSV
    fprintf(stderr, "%2d , %8d , %6.2lf , %6.2lf\n", NUMT, NUMTRIALS, 100. * probability, maxPerformance);
    #else
    fprintf(stderr, "%2d threads : %8d trials ; probability = %6.2f%% ; megatrials/sec = %6.2lf\n",
            NUMT, NUMTRIALS, 100. * probability, maxPerformance);
    #endif

    // Clean up
    delete[] vs;
    delete[] ths;
    delete[] gs;
    delete[] hs;
    delete[] ds;

    return 0;
}