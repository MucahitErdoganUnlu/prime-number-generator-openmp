#include <cstdio>
#include <cmath>
#include "omp.h"
#include <iostream>
#include <algorithm>

// Function declaration to check if a number is prime
int is_prime(int num, const int primes[], int count);

// Function declarations for parallel prime number generators with different OpenMP loop scheduling methods
double parallel_prime_generator(int, int, int, const char[]);
double guided_generator(int, int);
double dynamic_generator(int, int, int);
double static_generator(int, int, int);

int main() {
    // Arrays for different chunks and maximum primes
    int chunks[] = {5, 10, 50, 100, 200};
    int max_primes[] = {40, 400, 4000, 40000, 400000};
    
    // Array of schedule types
    char schedules[3][10] = {"static\0", "dynamic\0", "guided\0"};
    
    // File pointer for writing results to a CSV file
    FILE *fpt;

    // Open the results.csv file for writing
    fpt = fopen("results.csv", "w+");
    // Write the header for the CSV file
    fprintf(fpt, "M, OpenMP Loop Scheduling Method, Chunk Size, T1, T2, T4, T8, S2, S4, S8\n");

    // Loop over different maximum primes
    for (int max_prime : max_primes) {
        // Loop over different OpenMP loop scheduling methods
        for (auto &schedule : schedules) {
            // Loop over different chunk sizes
            for (int chunk : chunks) {
                // Measure time for different thread counts and write results to CSV file
                double T1 = parallel_prime_generator(1, max_prime, chunk, schedule);
                double T2 = parallel_prime_generator(2, max_prime, chunk, schedule);
                double T4 = parallel_prime_generator(4, max_prime, chunk, schedule);
                double T8 = parallel_prime_generator(8, max_prime, chunk, schedule);
                double S2 = T1 / T2;
                double S4 = T1 / T4;
                double S8 = T1 / T8;
                
                // Check if the schedule is guided and adjust the CSV output format accordingly
                if (schedule[0] == 'g')
                    fprintf(fpt, "%d, %s, %d, %f, %f, %f, %f, %f, %f, %f\n", max_prime, schedule, 0
                            , T1, T2, T4, T8, S2, S4, S8);
                else
                    fprintf(fpt, "%d, %s, %d, %f, %f, %f, %f, %f, %f, %f\n", max_prime, schedule, chunk
                            , T1, T2, T4, T8, S2, S4, S8);
            }
        }
    }
    // Close the results.csv file
    fclose(fpt);
}

// Function to check if a number is prime
int is_prime(int num, const int primes[], int count) {
    // Loop through the array of primes and check if num is divisible by any of them
    for (int i = 0; i < count; i++) {
        if (num % primes[i] == 0) {
            return 0; // Not a prime number
        }
    }
    return 1; // Prime number
}

// Function to call the appropriate parallel prime number generator based on the schedule type
double parallel_prime_generator(int num_threads, int M, int chunk, const char schedule[]) {
    // Choose the appropriate generator based on the schedule type
    if (schedule[0] == 'g')
        return guided_generator(num_threads, M);
    else if (schedule[0] == 'd')
        return dynamic_generator(num_threads, M, chunk);
    else
        return static_generator(num_threads, M, chunk);
}

// Guided prime number generator
double guided_generator(int num_threads, int M) {
    // Set the number of OpenMP threads
    omp_set_num_threads(num_threads);
    
    // Initialize array to store primes
    int p = M / 2;
    int primes[p];
    primes[0] = 2; // The first prime number

    int count = 1; // Number of primes found
    int num = 3;   // Starting from the next number

    int root = (int)sqrt(M);
    int n_of_small_primes;

    double start;
    double end;
    // Record the start time
    start = omp_get_wtime();

    // Generate small primes up to the square root of M
    while (num < root) {
        if (is_prime(num, primes, count)) {
            primes[count] = num;
            count++;
        }
        num += 2; // Check only odd numbers for primality
    }
    n_of_small_primes = count;

    int my_num = num;
    // Parallel loop using guided scheduling
    #pragma omp parallel for shared(count, primes, my_num, n_of_small_primes, M) private(num) default(none) schedule(guided)
    for (num = my_num; num <= M; num += 2) {
        if (is_prime(num, primes, n_of_small_primes)) {
            // Use a critical section to update the primes array
            #pragma omp critical
            {
                primes[count] = num;
                count++;
            }
        }
    }
    #pragma omp barrier

    primes[count] = 0; // last element of the prime list. used for determining the end of the array

    // Record the end time
    end = omp_get_wtime();
/*
    // Using the sort function from the <algorithm> header
    std::sort(primes, primes + count);

 
    // Print the sorted array
    std::cout << "Sorted Array: ";
    for (int i = 0; i < count; ++i) {
        std::cout << primes[i] << " ";
    }
*/
    return end - start;
}

// Dynamic prime number generator
double dynamic_generator(int num_threads, int M, int chunk) {
    // Set the number of OpenMP threads
    omp_set_num_threads(num_threads);
    
    // Initialize array to store primes
    int p = M / 2;
    int primes[p];
    primes[0] = 2; // The first prime number

    int count = 1; // Number of primes found
    int num = 3;   // Starting from the next number

    int root = (int)sqrt(M);
    int n_of_small_primes;

    double start;
    double end;
    // Record the start time
    start = omp_get_wtime();

    // Generate small primes up to the square root of M
    while (num < root) {
        if (is_prime(num, primes, count)) {
            primes[count] = num;
            count++;
        }
        num += 2; // Check only odd numbers for primality
    }
    n_of_small_primes = count;

    int my_num = num;
    // Parallel loop using dynamic scheduling with specified chunk size
    #pragma omp parallel for shared(count, primes, my_num, n_of_small_primes, M, chunk) private(num) default(none) schedule(dynamic, chunk)
        // Loop through odd numbers starting from where the sequential prime generator left off
    for (num = my_num; num <= M; num += 2) {
        // Check if the number is prime
        if (is_prime(num, primes, n_of_small_primes)) {
            // Use a critical section to update the primes array
            #pragma omp critical
            {
                primes[count] = num;
                count++;
            }
        }
    }
    // Synchronize all threads before moving on
    #pragma omp barrier

    primes[count] = 0; // Last element of the prime list. Used for determining the end of the array

    // Record the end time
    end = omp_get_wtime();

/*
    // Using the sort function from the <algorithm> header
    std::sort(primes, primes + count);

    // Print the sorted array
    std::cout << "Sorted Array: ";
    for (int i = 0; i < count; ++i) {
        std::cout << primes[i] << " ";
    }
*/
    return end - start;
}

// Static prime number generator
double static_generator(int num_threads, int M, int chunk) {
    // Set the number of OpenMP threads
    omp_set_num_threads(num_threads);

    // Initialize array to store primes
    int p = M / 2;
    int primes[p];
    primes[0] = 2; // The first prime number

    int count = 1; // Number of primes found
    int num = 3;   // Starting from the next number

    int root = (int)sqrt(M);
    int n_of_small_primes;

    double start;
    double end;
    // Record the start time
    start = omp_get_wtime();

    // Generate small primes up to the square root of M
    while (num < root) {
        if (is_prime(num, primes, count)) {
            primes[count] = num;
            count++;
        }
        num += 2; // Check only odd numbers for primality
    }
    n_of_small_primes = count;

    int my_num = num;
    // Parallel loop using static scheduling with specified chunk size
    #pragma omp parallel for shared(count, primes, my_num, n_of_small_primes, M, chunk) private(num) default(none) schedule(static, chunk)
    for (num = my_num; num <= M; num += 2) {
        // Check if the number is prime
        if (is_prime(num, primes, n_of_small_primes)) {
            // Use a critical section to update the primes array
            #pragma omp critical
            {
                primes[count] = num;
                count++;
            }
        }
    }
    // Synchronize all threads before moving on
    #pragma omp barrier

    primes[count] = 0; // Last element of the prime list. Used for determining the end of the array

    // Record the end time
    end = omp_get_wtime();

/*
    // Using the sort function from the <algorithm> header
    std::sort(primes, primes + count);

    // Print the sorted array
    std::cout << "Sorted Array: ";
    for (int i = 0; i < count; ++i) {
        std::cout << primes[i] << " ";
    }
*/
    return end - start;
}

       

