from collections import defaultdict
from data_access import fetch_all_data
from csp.solver import run_csp_balanced

data = fetch_all_data()
r = run_csp_balanced(data, max_spread=2)
print("status:", r["status"], "score:", r["score"])

req = {q["id"]: q for q in data["curriculum_requirements"]}
ta  = {t["id"]: t for t in data["teacher_assignments"]}
ts  = {s["id"]: s for s in data["timeslots"]}
grp = {g["id"]: g["group_name"] for g in data["student_groups"]}

by_cls_day = defaultdict(lambda: defaultdict(set))
for e in r.get("schedule_entries", []):
    gid = req[ta[e["tea_assignment_id"]]["cur_requirement_id"]]["student_group_id"]
    s = ts[e["timeslot_id"]]
    if s["day_of_week"] == 6:        # skip Friday
        continue
    by_cls_day[gid][s["day_of_week"]].add(s["hour_of_day"])

for gid, days in sorted(by_cls_day.items()):
    counts = [len(h) for h in days.values()]
    if len(counts) < 2:
        continue
    spread = max(counts) - min(counts)
    flag = "   <-- מעל 2!" if spread > 2 else ""
    print(f"{grp.get(gid, gid)}: ספירות={sorted(counts)} פער={spread}{flag}")
