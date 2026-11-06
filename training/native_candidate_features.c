/* Isolated prototype. Inputs are checked before any index or shift. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ACTIONS 1840
#define FEATURES 16
#define STRIDE 13

const char *tetris_features_abi(void) {
    return "tetris-features-v1:u16[20]:i64[n,13]:i32[16]:u8[1840,16]:u8[1840]";
}

static int height(uint32_t c) { return c ? 32 - __builtin_clz(c) : 0; }
static int maximum(int a, int b) { return a > b ? a : b; }
static int minimum(int a, int b) { return a < b ? a : b; }

static void columns_from_rows(const uint16_t rows[20], uint32_t columns[10]) {
    memset(columns, 0, 10 * sizeof(*columns));
    for (int y = 0; y < 20; ++y)
        for (int x = 0; x < 10; ++x)
            if (rows[y] & (1u << x)) columns[x] |= 1u << (19 - y);
}

/* Nonzero means unsupported input: wrapper retries the original Python function
 * to preserve its exception type and ordering. Partial outputs are discarded. */
int tetris_candidate_features(const uint16_t rows[20], const int64_t *placements,
                             size_t count, int current, int held,
                             const int32_t scales[16], uint8_t *matrix, uint8_t *mask) {
    if (current < 0 || current > 6 || held < 0 || held > 6) return 1;
    for (int y = 0; y < 20; ++y) if (rows[y] > 1023) return 2;
    uint32_t base_columns[10];
    columns_from_rows(rows, base_columns);
    int before_holes = 0, before_height = 0;
    for (int x = 0; x < 10; ++x) {
        int h = height(base_columns[x]);
        before_height = maximum(before_height, h);
        before_holes += h - __builtin_popcount(base_columns[x]);
    }
    memset(matrix, 0, ACTIONS * FEATURES);
    memset(mask, 0, ACTIONS);
    for (size_t index = 0; index < count; ++index) {
        const int64_t *p = placements + index * STRIDE;
        if (p[0] < -ACTIONS || p[0] >= ACTIONS || p[1] < 0 || p[1] > 1 ||
            p[2] < 0 || p[2] > 3 || p[3] < 0 || p[3] > 9 || p[4] < -3 || p[4] > 19)
            return 3;
        int action = (int)(p[0] < 0 ? p[0] + ACTIONS : p[0]);
        if (mask[action]) return 4;
        mask[action] = 1;
        uint16_t placed_rows[20];
        uint32_t columns[10];
        memcpy(placed_rows, rows, sizeof(placed_rows));
        memcpy(columns, base_columns, sizeof(columns));
        int visible = 0, cleared = 0;
        for (int cell = 0; cell < 4; ++cell) {
            int64_t x = p[5 + 2 * cell], y = p[6 + 2 * cell];
            if (y < 0) continue;
            if (x < 0 || x >= 10 || y >= 20) return 5;
            columns[x] |= 1u << (19 - y);
            placed_rows[y] |= (uint16_t)(1u << x);
            visible = 1;
        }
        if (visible) {
            int target = 19;
            for (int y = 19; y >= 0; --y) {
                if (placed_rows[y] == 1023) ++cleared;
                else placed_rows[target--] = placed_rows[y];
            }
            if (cleared) {
                while (target >= 0) placed_rows[target--] = 0;
                columns_from_rows(placed_rows, columns);
            }
        }
        int heights[10], holes = 0, aggregate = 0, max_height = 0, top_four = 0;
        for (int x = 0; x < 10; ++x) {
            heights[x] = height(columns[x]);
            holes += heights[x] - __builtin_popcount(columns[x]);
            aggregate += heights[x];
            max_height = maximum(max_height, heights[x]);
            top_four += __builtin_popcount(columns[x] >> 16);
        }
        int bumpiness = 0, wells = 0;
        for (int x = 0; x < 10; ++x) {
            if (x) {
                int delta = heights[x] - heights[x-1];
                bumpiness += delta < 0 ? -delta : delta;
            }
            int left = x ? heights[x-1] : 20;
            int right = x < 9 ? heights[x+1] : 20;
            wells += maximum(0, minimum(left, right) - heights[x]);
        }
        int values[16] = {p[1] ? held : current, (int)p[1], (int)p[2], (int)p[3],
            (int)p[4]+3, cleared, holes, aggregate, max_height, bumpiness, wells,
            maximum(0, holes-before_holes), maximum(0, before_holes-holes),
            before_height, top_four, !visible};
        for (int f = 0; f < FEATURES; ++f) {
            if (values[f] < 0 || values[f] > scales[f] || values[f] > 255) return 6;
            matrix[action * FEATURES + f] = (uint8_t)values[f];
        }
    }
    return 0;
}
