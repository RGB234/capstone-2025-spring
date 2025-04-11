from enum import Enum


class ListResponse(Enum):
    target = "target"  # 검색 대상
    public_number = "공포번호"  # 공포번호
    keyword = "키워드"  # 검색어
    section = "section"  # 검색범위 (EvtNm: 판례명 / bdyText: 본문)
    totalCnt = "totalCnt"  # 검색결과 개수
    page = "page"  # 출력 페이지
    prec_id = "prec id"  # 검색결과 번호
    case_reference_number = "판례일련번호"  # 판례일련번호
    case_name = "사건명"  # 사건명
    case_number = "사건번호"  # 사건번호
    ruling_date = "선고일자"  # 선고일자
    court_name = "법원명"  # 법원명
    court_type_code = "법원종류코드"  # 법원종류코드 (대법원: 400201, 하위법원: 400202)
    case_type_name = "사건종류명"  # 사건종류명
    case_type_code = "사건종류코드"  # 사건종류코드
    judgment_type = "판결유형"  # 판결유형
    ruling = "선고"  # 선고
    data_source_name = "데이터출처명"  # 데이터출처명
    case_detail_link = "판례상세링크"  # 판례 상세 링크


from enum import Enum


class DetailResponse(Enum):
    case_info_id = "판례정보일련번호"  # 판례정보일련번호
    case_name = "사건명"  # 사건명
    case_number = "사건번호"  # 사건번호
    ruling_date = "선고일자"  # 선고일자
    ruling = "선고"  # 선고
    court_name = "법원명"  # 법원명
    court_type_code = "법원종류코드"  # 법원종류코드 (대법원: 400201, 하위법원: 400202)
    case_type_name = "사건종류명"  # 사건종류명
    case_type_code = "사건종류코드"  # 사건종류코드
    judgment_type = "판결유형"  # 판결유형
    judgment_summary = "판시사항"  # 판시사항
    ruling_basis = "판결요지"  # 판결요지
    reference_provision = "참조조문"  # 참조조문
    reference_case = "참조판례"  # 참조판례
    case_details = "판례내용"  # 판례내용
