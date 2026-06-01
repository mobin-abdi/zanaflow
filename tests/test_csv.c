#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <zanaflow/data/csv_loader.h>

static void write_file(const char *path, const char *text) {
    FILE *fp = fopen(path, "w");
    assert(fp != NULL);
    fputs(text, fp);
    fclose(fp);
}

static void test_label_last_with_header(void) {
    const char *path = "test.csv";
    write_file(path,
        "x1,x2,y\n"
        "1.0,2.0,0\n"
        "3.0,4.0,1\n"
        "5.0,6.0,0\n"
    );

    zf_csv_data_f32 data;
    int rc = zf_csv_load_f32(path, 2, 1, 1, &data);
    assert(rc == 0);
    assert(data.n == 3);
    assert(data.d == 2);

    assert(data.X[0] == 1.0f); assert(data.X[1] == 2.0f); assert(data.y[0] == 0.0f);
    assert(data.X[2] == 3.0f); assert(data.X[3] == 4.0f); assert(data.y[1] == 1.0f);
    assert(data.X[4] == 5.0f); assert(data.X[5] == 6.0f); assert(data.y[2] == 0.0f);

    zf_csv_data_free(&data);
    remove(path);
}

static void test_label_first_no_header(void) {
    const char *path = "tmp_test_label_first.csv";
    write_file(path,
        "0,1.0,2.0\n"
        "1,3.0,4.0\n"
    );

    zf_csv_data_f32 data;
    int rc = zf_csv_load_f32(path, 2, 0, 0, &data);
    assert(rc == 0);
    assert(data.n == 2);
    assert(data.d == 2);

    assert(data.y[0] == 0.0f);
    assert(data.X[0] == 1.0f);
    assert(data.X[1] == 2.0f);

    assert(data.y[1] == 1.0f);
    assert(data.X[2] == 3.0f);
    assert(data.X[3] == 4.0f);

    zf_csv_data_free(&data);
    remove(path);
}

static void test_malformed_row_should_fail(void) {
    const char *path = "tmp_test_bad.csv";
    write_file(path,
        "x1,x2,y\n"
        "1.0,2.0,0\n"
        "3.0,4.0\n"
    );

    zf_csv_data_f32 data;
    int rc = zf_csv_load_f32(path, 2, 1, 1, &data);
    assert(rc != 0);

    zf_csv_data_free(&data);
    remove(path);
}

int main(void) {
    test_label_last_with_header();
    test_label_first_no_header();
    test_malformed_row_should_fail();

    printf("test_csv passed\n");
    return 0;
}
