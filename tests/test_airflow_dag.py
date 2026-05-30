def test_dag_imports_without_error():
    """
    Import the DAG module and verify it is a valid DAG object.

    This test catches syntax errors, bad imports at module level,
    and missing DAG configuration before the file reaches Airflow.
    If this test fails, Airflow would show a broken DAG in the UI.
    """
    from dags.pharmacovigilance_pipeline import dag

    assert dag is not None
    assert dag.dag_id == "pharmacovigilance_pipeline"


def test_dag_has_correct_number_of_tasks():
    from dags.pharmacovigilance_pipeline import dag

    assert len(dag.tasks) == 5


def test_dag_task_ids_are_correct():
    from dags.pharmacovigilance_pipeline import dag

    expected = {
        "ingest_openfda_data",
        "bronze_to_silver",
        "data_quality_checks",
        "build_gold_tables",
        "load_gold_to_bigquery",
    }
    actual = {task.task_id for task in dag.tasks}
    assert actual == expected


def test_dag_task_dependencies_are_correct():
    """
    Verify the task chain: ingest → silver → quality → gold → bigquery.

    Each task must have exactly one downstream task (except the last).
    This ensures the pipeline cannot run gold before quality passes,
    or load to BigQuery before gold is built.
    """
    from dags.pharmacovigilance_pipeline import dag

    task_map = {task.task_id: task for task in dag.tasks}

    assert task_map["bronze_to_silver"] in task_map["ingest_openfda_data"].downstream_list
    assert task_map["data_quality_checks"] in task_map["bronze_to_silver"].downstream_list
    assert task_map["build_gold_tables"] in task_map["data_quality_checks"].downstream_list
    assert task_map["load_gold_to_bigquery"] in task_map["build_gold_tables"].downstream_list


def test_dag_schedule_is_monthly():
    from dags.pharmacovigilance_pipeline import dag

    assert dag.schedule_interval == "0 3 1 * *"


def test_dag_catchup_is_disabled():
    from dags.pharmacovigilance_pipeline import dag

    assert dag.catchup is False


def test_dag_has_retry_configuration():
    from dags.pharmacovigilance_pipeline import dag

    for task in dag.tasks:
        assert task.retries == 2
