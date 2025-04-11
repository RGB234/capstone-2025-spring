from enum import Enum


class ListParameter(Enum):
    OC = "OC"  # 사용자 이메일 ID (예: g4c@korea.kr → OC=g4c) [필수]
    target = "target"  # 서비스 대상 (예: prec) [필수]
    type = "type"  # 출력 형식: HTML/XML/JSON
    search = "search"  # 검색범위 (1: 판례명, 2: 본문검색)
    query = "query"  # 검색어 (검색결과 리스트에서 원하는 질의어)
    display = "display"  # 검색된 결과 개수 (기본값=20, 최대=100)
    page = "page"  # 검색 결과 페이지 (기본값=1)
    org = "org"  # 법원종류 (대법원: 400201, 하위법원: 400202)
    curt = "curt"  # 법원명 (대법원, 서울고등법원, 광주지법, 인천지방법원 등)
    JO = "JO"  # 참조법령명 (형법, 민법 등)
    gana = "gana"  # 사전식 검색 (ga, na, da, …)
    sort = "sort"  # 정렬 옵션: lacs (사건명 오름차순), ldes (사건명 내림차순),
    # dasc (선고일자 오름차순), ddes (선고일자 내림차순, 기본),
    # nasc (법원명 오름차순), ndes (법원명 내림차순)
    date = "date"  # 판례 선고일자
    prncYd = "prncYd"  # 선고일자 검색 범위 (예: 20090101~20090130)
    nb = "nb"  # 판례 사건번호
    datSrcNm = (
        "datSrcNm"  # 데이터 출처 (국세법령정보시스템, 근로복지공단산재판례, 대법원)
    )
    popYn = "popYn"  # 상세화면 팝업창 여부 (팝업창으로 띄우려면 'popYn=Y')


class DetailParameter(Enum):
    OC = "OC"  # 사용자 이메일 ID (예: g4c@korea.kr → OC=g4c) [필수]
    target = "target"  # 서비스 대상 (예: prec) [필수]
    type = "type"  # 출력 형식: HTML/XML/JSON (*국세청 판례 본문 조회는 HTML만 가능)
    ID = "ID"  # 판례 일련번호 [필수]
    LM = "LM"  # 판례명
