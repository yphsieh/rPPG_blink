### Data Structure
```
data.py
|
|---Subject(class)
|---SubjectGroup(class)
```
1. data.py裡面有一個Subject的class，每個受試者都是一個Subject的object。
2. SubjectGroup是另一個class，會把每個受試者對應到的subject存到裡面。
```python
# 讀檔
subject_group = SubjectGroup(pulse_dir=args.pulse_data_dir, label_filename='label.xlsx')
# 得到pad過的data
data = subject_group.get_pad_pulse_data(pad_pulse_length=500)
```