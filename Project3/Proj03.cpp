#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string>

// setting the number of threads:
#ifndef NUMT
#define NUMT 1
#endif

// setting the number of capitals we want to try:
#ifndef NUMCAPITALS
#define NUMCAPITALS 5
#endif

// maximum iterations to allow looking for convergence:
#define MAXITERATIONS 100

// how many tries to discover the maximum performance:
#define NUMTRIES 30

struct city
{
    std::string name;
    float longitude;
    float latitude;
    int capitalnumber;
    float mindistance;
};

#include "UsCities.data"

// setting the number of cities we want to try:
#define NUMCITIES (sizeof(Cities) / sizeof(struct city))

struct capital
{
    std::string name;
    float longitude;
    float latitude;
    float longsum;
    float latsum;
    int numsum;
};

struct capital Capitals[NUMCAPITALS];

float Distance(int city, int capital)
{
    float dx = Cities[city].longitude - Capitals[capital].longitude;
    float dy = Cities[city].latitude - Capitals[capital].latitude;
    return sqrtf(dx * dx + dy * dy);
}

int main(int argc, char *argv[])
{
#ifdef _OPENMP
    fprintf(stderr, "OpenMP is supported -- version = %d\n", _OPENMP);
#else
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

    omp_set_num_threads(NUMT); // set the number of threads to use in parallelizing the for-loop

    // seed the capitals:
    // (this is just picking initial capital cities at uniform intervals)
    for (int k = 0; k < NUMCAPITALS; k++)
    {
        int cityIndex = k * (NUMCITIES - 1) / (NUMCAPITALS - 1);
        Capitals[k].longitude = Cities[cityIndex].longitude;
        Capitals[k].latitude = Cities[cityIndex].latitude;
    }

    double maxMegaCityCapitalsPerSecond = 0.;

    for (int n = 0; n < MAXITERATIONS; n++)
    {
        // reset the summations for the capitals:
        for (int k = 0; k < NUMCAPITALS; k++)
        {
            Capitals[k].longsum = 0.;
            Capitals[k].latsum = 0.;
            Capitals[k].numsum = 0;
        }

        // Run NUMTRIES trials to find the maximum performance
        double totalTime = 0.;
        for (int t = 0; t < NUMTRIES; t++)
        {
            double time0 = omp_get_wtime();

            // Per-thread arrays for accumulation
            float **thread_longsum = new float*[NUMT];
            float **thread_latsum = new float*[NUMT];
            int **thread_numsum = new int*[NUMT];
            for (int tid = 0; tid < NUMT; tid++)
            {
                thread_longsum[tid] = new float[NUMCAPITALS]();
                thread_latsum[tid] = new float[NUMCAPITALS]();
                thread_numsum[tid] = new int[NUMCAPITALS]();
            }

            // Parallel loop to assign cities to nearest capitals
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                #pragma omp for
                for (int i = 0; i < NUMCITIES; i++)
                {
                    int capitalnumber = -1;
                    float mindistance = 1.e+37;

                    for (int k = 0; k < NUMCAPITALS; k++)
                    {
                        float dist = Distance(i, k);
                        if (dist < mindistance)
                        {
                            mindistance = dist;
                            capitalnumber = k;
                        }
                    }

                    // Assign city to closest capital
                    Cities[i].capitalnumber = capitalnumber;

                    // Accumulate in thread-local arrays
                    thread_longsum[tid][capitalnumber] += Cities[i].longitude;
                    thread_latsum[tid][capitalnumber] += Cities[i].latitude;
                    thread_numsum[tid][capitalnumber]++;
                }
            }

            // Merge thread-local arrays into Capitals
            for (int tid = 0; tid < NUMT; tid++)
            {
                for (int k = 0; k < NUMCAPITALS; k++)
                {
                    Capitals[k].longsum += thread_longsum[tid][k];
                    Capitals[k].latsum += thread_latsum[tid][k];
                    Capitals[k].numsum += thread_numsum[tid][k];
                }
                delete[] thread_longsum[tid];
                delete[] thread_latsum[tid];
                delete[] thread_numsum[tid];
            }
            delete[] thread_longsum;
            delete[] thread_latsum;
            delete[] thread_numsum;

            double time1 = omp_get_wtime();
            totalTime += (time1 - time0);
        }

        // Calculate average time per trial
        double avgTime = totalTime / NUMTRIES;
        double megaCityCapitalsPerSecond = (double)NUMCITIES * (double)NUMCAPITALS / avgTime / 1000000.;

        // Update maximum performance
        if (megaCityCapitalsPerSecond > maxMegaCityCapitalsPerSecond)
            maxMegaCityCapitalsPerSecond = megaCityCapitalsPerSecond;

        // Update capital locations (average longitude and latitude)
        for (int k = 0; k < NUMCAPITALS; k++)
        {
            if (Capitals[k].numsum > 0)
            {
                Capitals[k].longitude = Capitals[k].longsum / Capitals[k].numsum;
                Capitals[k].latitude = Capitals[k].latsum / Capitals[k].numsum;
            }
        }
    }

    // Extra Credit: Find the closest city to each capital
    for (int k = 0; k < NUMCAPITALS; k++)
    {
        float minDistance = 1.e+37;
        int closestCity = -1;

        for (int i = 0; i < NUMCITIES; i++)
        {
            float dist = Distance(i, k);
            if (dist < minDistance)
            {
                minDistance = dist;
                closestCity = i;
            }
        }

        Capitals[k].name = Cities[closestCity].name; // Save name of closest city

        // Print to stderr so it shows up in terminal, not CSV
        fprintf(stderr, "Capital %d is closest to city: %s\n",
                k, Cities[closestCity].name.c_str());
    }

    // Print the longitude-latitude of each new capital city for NUMT == 1
    if (NUMT == 1)
    {
        for (int k = 0; k < NUMCAPITALS; k++)
        {
            fprintf(stderr, "\t%3d:  %8.2f , %8.2f\n", k, Capitals[k].longitude, Capitals[k].latitude);
        }
    }

    printf("%d\t%zu\t%d\t%.3lf\n", NUMT, NUMCITIES, NUMCAPITALS, maxMegaCityCapitalsPerSecond);

    return 0;
}