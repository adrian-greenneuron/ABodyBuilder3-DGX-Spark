"""Select nested distinct panels from retained CSVs; never invent chain pairs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",type=Path,required=True);parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    columns=["name","heavy","light"]
    first=[{key:row[key] for key in columns} for row in csv.DictReader(args.benchmark.open())]
    assert len(first)==8
    pairs={(row["heavy"],row["light"]) for row in first};selected=first[:]
    for row in csv.DictReader(args.native.open()):
        key=(row["heavy"],row["light"])
        if key not in pairs:
            pairs.add(key);selected.append({"name":f"native_{len(selected)-8:06d}","heavy":key[0],"light":key[1]})
        if len(selected)==128:break
    assert len(pairs)==128 and len({row["name"] for row in selected})==128
    args.output.mkdir(exist_ok=True,parents=True);records={}
    for count in [64,128]:
        path=args.output/f"antibodies_{count}.csv"
        with path.open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n");writer.writeheader();writer.writerows(selected[:count])
        lengths=[len(row["heavy"])+len(row["light"]) for row in selected[:count]]
        records[path.name]={"count":count,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"distinct_pairs":count,"combined_length_range":[min(lengths),max(lengths)]}
    (args.output/"selection_manifest_261026.json").write_text(json.dumps({"inputs":records,
        "source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.benchmark,args.native]},
        "selection":"Eight retained benchmark pairs, then first distinct pairs from the frozen study-interleaved OAS panel; native partners preserved."},indent=2)+"\n")


if __name__=="__main__":main()
