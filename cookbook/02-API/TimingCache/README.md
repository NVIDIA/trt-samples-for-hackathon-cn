#

## Steps to run

```shell
python3 main.py
```

## Output for reference: ./result-*.log

+ Using timing cache across device with different SM is not allowed, 

```txt
[03/31/2023-07:12:49] [TRT] [E] 4: Timing cache header mismatch:Incoming ITimingCache: UUID = GPU-e95b6d5c-6b23-2af9-45f9-3df237cfd30f, commit = 9a73423585f0fcfc
Runtime device: UID = GPU-b98ca71b-0733-40f1-a3e7-ddff10b5056f, commit = 5a515d811b81ec93
```

