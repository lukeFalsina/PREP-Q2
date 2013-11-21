
// Allocate the arbitrary infinity value in constant memory
__constant__ int infinity;

/**
 * This device function returns the binary entropy value computed
 * for a variable whose possible outcomes could only be success
 * or failure.
 *
 * \param	x		[IN] - A probability value for the success case (0 <= x <= 1)
 *
 * \return	binary entropy value of x
 */
__device__ float entropy(float x) {
	return -x * log2f(x) - (1 - x) * log2f(1-x);
}

/**
 * This kernel simply computes the value of the amount of rows, n,
 * by using the following formula and symbols:
 *
 * n = k * (1 - (1 / (log_2(1 - delta)))) / (1 - H(delta))
 *
 * n = amount of rows required by the error correcting code
 * k = a statistical security parameter
 * delta = relative minimum distance of the random matrix M (0 <= delta <= 1)
 * H(.) = Shannon entropy function for a Bernulli variable
 *
 * In particular the parameter k is chosen on demand and could be
 * omitted from this computation and delta is the input parameter.
 *
 * \param	amountRowsResults		[OUT] - An array which contains all the results of the computation
 * \param	sensibility				[IN] - It represents the numbers of intervals tested for delta
 * \param	maxDeltaThreshold		[IN] - It specifies the maximum assignable value to delta
 */
__global__ void computeAmountRows(float *amountRowsResults, int sensibility, float maxDeltaThreshold)
{

	// Calculate the index of the current thread
	const unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// Compute the value of the tested input parameter delta
	float delta = (float) threadId / sensibility + (float) 1/sensibility;

	if(delta <= maxDeltaThreshold) {

		// The amount of rows needed for the selected delta is stored at position threadId
		//amountRowsResults[threadId] = (1 - (1 / (log2f(1 - delta)))) / (1 - entropy(delta));
		//amountRowsResults[threadId] = threadId + 10.0f;
		float numerator = 1 - ( 1 / log2f(1 - delta) );
		float denominator = 1 - entropy(delta);
		amountRowsResults[threadId] = numerator / denominator;
	}
	else {
		// This value is out of bound and should not be considered
		amountRowsResults[threadId] = infinity;
	}
}

