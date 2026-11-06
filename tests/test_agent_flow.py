async def _events(agent, df):
    return [e async for e in agent.run_streaming("Compute the mean of x column", dataframe=df)]


async def test_full_run(make_agent, df):
    agent = make_agent()
    ev = await _events(agent, df)
    types = [e.event_type for e in ev]
    assert types[-1] == "complete" and "error" not in types
    assert "step_complete" in types


async def test_self_correction_trail(make_agent, df):
    agent = make_agent(code_writes=["print(df['wrong'].mean())"], code_fixes=["print(df['x'].mean())"])
    ev = await _events(agent, df)
    step = next(e for e in ev if e.event_type == "step_complete" and e.data.get("corrections"))
    c = step.data["corrections"][0]
    assert c["error_type"] == "KeyError"
    assert "wrong" in c["failed_code"] and "'x'" in c["fixed_code"]
    assert c["verified"] is True
    assert "wrong" in step.data["original_code"]


async def test_unverified_fix_flagged(make_agent, df):
    agent = make_agent(code_writes=["print(df['a'])"], code_fixes=["print(df['b'])", "print(df['c'])"])
    ev = await _events(agent, df)
    steps = [e for e in ev if e.event_type == "step_complete" and e.data.get("corrections")]
    assert steps and not all(c["verified"] for c in steps[0].data["corrections"])


async def test_blocked_code_reports_security_error(make_agent, df):
    agent = make_agent(code_writes=["__import__('os').system('id')"], code_fixes=[])
    ev = await _events(agent, df)
    step = next(e for e in ev if e.event_type == "step_complete" and e.data.get("step_id") == "s1")
    assert step.data.get("error_type") == "SecurityError"
