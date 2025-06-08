# 👋&nbsp; Hi there!

고려대 통계학과 데이터애널리틱스특수연구3 프로젝트~

## Requirements

- python >= 3.10
- Airflow = 2.10.5

## Usage
<code>src/config</code> 디렉터리에 있는 숨김파일 <code>.env.example</code> 파일을 <code>.env</code> 파일로 복사해서 OpenAI api key, reddit api 정보 넣기
```bash
docker compose up -d
```

### Notebook data paths

The LSTM and Dash notebooks load their CSV files from `notebook/data` when
the directory exists, otherwise they fall back to a top-level `data`
directory. This allows them to run from either the repository root or the
`notebook` directory without modification.



