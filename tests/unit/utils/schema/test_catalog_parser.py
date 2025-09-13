"""
Catalog Parser 테스트 (Model Discovery & 카탈로그 파싱 핵심 모듈)
tests/README.md 전략 준수: 컨텍스트 기반, 퍼블릭 API, 실제 객체, 결정론적

테스트 대상 핵심 기능:
- load_model_catalog() - src/models/catalog.yaml 파일 로드 및 파싱
- 카탈로그 파싱과 검증
- 에러 핸들링 (파일 없음, 잘못된 형식)
- 경로 처리 (상대 경로, 절대 경로)

핵심 Edge Cases:
- catalog.yaml 파일이 존재하지 않는 경우
- 잘못된 YAML 형식
- 빈 카탈로그 파일
- 권한 문제로 읽기 실패
- 다양한 카탈로그 구조
"""
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open, Mock
import tempfile
import os

from src.utils.schema.catalog_parser import load_model_catalog


class TestLoadModelCatalog:
    """load_model_catalog 함수 핵심 테스트"""
    
    def test_load_model_catalog_success_with_valid_yaml(self):
        """케이스 A: 올바른 catalog.yaml 파일 로드 성공"""
        # Given: 유효한 카탈로그 내용
        valid_catalog = {
            'classification': [
                {
                    'name': 'RandomForest',
                    'module': 'sklearn.ensemble',
                    'class': 'RandomForestClassifier',
                    'data_handler': 'classification'
                },
                {
                    'name': 'XGBoost',
                    'module': 'xgboost',
                    'class': 'XGBClassifier',
                    'data_handler': 'classification'
                }
            ],
            'regression': [
                {
                    'name': 'LinearRegression',
                    'module': 'sklearn.linear_model',
                    'class': 'LinearRegression',
                    'data_handler': 'regression'
                }
            ],
            'timeseries': [
                {
                    'name': 'LSTM',
                    'module': 'tensorflow.keras.models',
                    'class': 'Sequential',
                    'data_handler': 'deeplearning'
                }
            ]
        }
        
        valid_yaml_content = yaml.dump(valid_catalog)
        
        # When: mock을 사용하여 파일 읽기 시뮬레이션
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=valid_yaml_content):
            
            result = load_model_catalog()
        
        # Then: 올바른 카탈로그 구조 반환
        assert isinstance(result, dict)
        assert 'classification' in result
        assert 'regression' in result
        assert 'timeseries' in result
        
        # Classification 모델 검증
        assert len(result['classification']) == 2
        assert result['classification'][0]['name'] == 'RandomForest'
        assert result['classification'][1]['name'] == 'XGBoost'
        
        # Regression 모델 검증
        assert len(result['regression']) == 1
        assert result['regression'][0]['name'] == 'LinearRegression'
        
        # Timeseries 모델 검증
        assert len(result['timeseries']) == 1
        assert result['timeseries'][0]['name'] == 'LSTM'
        assert result['timeseries'][0]['data_handler'] == 'deeplearning'
    
    def test_load_model_catalog_file_not_exists(self):
        """케이스 B: catalog.yaml 파일이 존재하지 않는 경우"""
        # Given: 파일이 존재하지 않음
        with patch('pathlib.Path.exists', return_value=False):
            
            # When: 카탈로그 로드 시도
            result = load_model_catalog()
            
            # Then: 빈 딕셔너리 반환 (에러 없이)
            assert result == {}
    
    def test_load_model_catalog_invalid_yaml_format(self):
        """케이스 C: 잘못된 YAML 형식으로 인한 파싱 에러"""
        # Given: 잘못된 YAML 내용
        invalid_yaml_content = """
        classification:
            - name: RandomForest
              module: sklearn.ensemble
            invalid_structure: [
                missing_closing_bracket
        regression:
            - name: "unclosed_quote
        """
        
        # When: 잘못된 YAML 파일 로드 시도
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=invalid_yaml_content):
            
            result = load_model_catalog()
            
            # Then: 예외 처리로 빈 딕셔너리 반환
            assert result == {}
    
    def test_load_model_catalog_empty_file(self):
        """케이스 D: 빈 카탈로그 파일"""
        # Given: 빈 파일 내용
        empty_content = ""
        
        # When: 빈 파일 로드
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=empty_content):
            
            result = load_model_catalog()
            
            # Then: None 또는 빈 딕셔너리 반환 (yaml.safe_load가 None 반환할 수 있음)
            assert result is None or result == {}
    
    def test_load_model_catalog_yaml_loads_none(self):
        """케이스 E: YAML 로드가 None을 반환하는 경우"""
        # Given: yaml.safe_load가 None 반환하는 경우 (빈 파일 등)
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=""), \
             patch('yaml.safe_load', return_value=None):
            
            # When: 카탈로그 로드
            result = load_model_catalog()
            
            # Then: None이 반환됨 (정상 동작)
            assert result is None
    
    def test_load_model_catalog_permission_error(self):
        """케이스 F: 파일 읽기 권한 문제"""
        # Given: 권한 문제로 파일 읽기 실패
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', side_effect=PermissionError("Permission denied")):
            
            # When: 권한 에러 발생 시
            result = load_model_catalog()
            
            # Then: 예외 처리로 빈 딕셔너리 반환
            assert result == {}
    
    def test_load_model_catalog_file_read_error(self):
        """케이스 G: 기타 파일 읽기 에러"""
        # Given: 파일 읽기 중 다른 에러 발생
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', side_effect=IOError("Disk full")):
            
            # When: I/O 에러 발생 시
            result = load_model_catalog()
            
            # Then: 예외 처리로 빈 딕셔너리 반환
            assert result == {}
    
    def test_load_model_catalog_minimal_valid_structure(self):
        """케이스 H: 최소한의 유효한 카탈로그 구조"""
        # Given: 최소한의 유효한 카탈로그
        minimal_catalog = {
            'classification': [
                {
                    'name': 'SimpleModel',
                    'module': 'sklearn.dummy',
                    'class': 'DummyClassifier'
                }
            ]
        }
        
        minimal_yaml_content = yaml.dump(minimal_catalog)
        
        # When: 최소한의 카탈로그 로드
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=minimal_yaml_content):
            
            result = load_model_catalog()
        
        # Then: 올바르게 로드됨
        assert isinstance(result, dict)
        assert 'classification' in result
        assert len(result['classification']) == 1
        assert result['classification'][0]['name'] == 'SimpleModel'
    
    def test_load_model_catalog_complex_nested_structure(self):
        """케이스 I: 복잡한 중첩 구조의 카탈로그"""
        # Given: 복잡한 중첩 구조
        complex_catalog = {
            'classification': [
                {
                    'name': 'RandomForest',
                    'module': 'sklearn.ensemble',
                    'class': 'RandomForestClassifier',
                    'data_handler': 'classification',
                    'hyperparameters': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [5, 10, 15],
                        'min_samples_split': [2, 5, 10]
                    },
                    'preprocessing': {
                        'scaling': 'standard',
                        'encoding': 'onehot'
                    },
                    'supported_tasks': ['binary', 'multiclass'],
                    'requirements': ['scikit-learn>=1.0.0']
                }
            ],
            'clustering': [
                {
                    'name': 'KMeans',
                    'module': 'sklearn.cluster',
                    'class': 'KMeans',
                    'data_handler': 'clustering',
                    'hyperparameters': {
                        'n_clusters': [3, 5, 8, 10],
                        'init': ['k-means++', 'random'],
                        'max_iter': [300, 500]
                    }
                }
            ]
        }
        
        complex_yaml_content = yaml.dump(complex_catalog)
        
        # When: 복잡한 카탈로그 로드
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=complex_yaml_content):
            
            result = load_model_catalog()
        
        # Then: 중첩 구조가 올바르게 파싱됨
        assert isinstance(result, dict)
        assert 'classification' in result
        assert 'clustering' in result
        
        # 중첩된 hyperparameters 검증
        rf_model = result['classification'][0]
        assert 'hyperparameters' in rf_model
        assert 'n_estimators' in rf_model['hyperparameters']
        assert rf_model['hyperparameters']['n_estimators'] == [100, 200, 300]
        
        # 중첩된 preprocessing 검증
        assert 'preprocessing' in rf_model
        assert rf_model['preprocessing']['scaling'] == 'standard'
        
        # 배열 구조 검증
        assert 'supported_tasks' in rf_model
        assert 'binary' in rf_model['supported_tasks']
        assert 'multiclass' in rf_model['supported_tasks']
    
    def test_load_model_catalog_path_resolution(self):
        """케이스 J: 경로 해석 테스트 (실제 파일 시스템과의 상호작용)"""
        # Given: 임시 디렉토리와 파일 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # 가상의 프로젝트 구조 생성
            src_dir = Path(temp_dir) / "src"
            models_dir = src_dir / "models"
            utils_dir = src_dir / "utils" / "schema"
            
            models_dir.mkdir(parents=True)
            utils_dir.mkdir(parents=True)
            
            # catalog.yaml 파일 생성
            catalog_file = models_dir / "catalog.yaml"
            test_catalog = {
                'test_models': [
                    {'name': 'TestModel', 'module': 'test.module', 'class': 'TestClass'}
                ]
            }
            catalog_file.write_text(yaml.dump(test_catalog))
            
            # catalog_parser.py의 __file__ 경로를 임시 디렉토리로 mock
            fake_parser_file = utils_dir / "catalog_parser.py"
            fake_parser_file.write_text("# fake parser file")
            
            # When: 실제 경로 해석으로 카탈로그 로드
            with patch('src.utils.schema.catalog_parser.__file__', str(fake_parser_file)):
                result = load_model_catalog()
            
            # Then: 경로가 올바르게 해석되어 카탈로그 로드됨
            assert isinstance(result, dict)
            assert 'test_models' in result
            assert result['test_models'][0]['name'] == 'TestModel'
    
    def test_load_model_catalog_unicode_content(self):
        """케이스 K: 유니코드 내용이 포함된 카탈로그"""
        # Given: 유니코드 문자가 포함된 카탈로그
        unicode_catalog = {
            'classification': [
                {
                    'name': 'RandomForest한글',
                    'description': '랜덤 포레스트 분류기 🌲',
                    'module': 'sklearn.ensemble',
                    'class': 'RandomForestClassifier',
                    'author': 'ML팀',
                    'tags': ['기계학습', '앙상블', '🚀'],
                    'notes': 'This model supports 한글 and emojis 😊'
                }
            ]
        }
        
        unicode_yaml_content = yaml.dump(unicode_catalog, allow_unicode=True)
        
        # When: 유니코드 카탈로그 로드
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=unicode_yaml_content):
            
            result = load_model_catalog()
        
        # Then: 유니코드 내용이 올바르게 로드됨
        assert isinstance(result, dict)
        assert 'classification' in result
        
        model = result['classification'][0]
        assert model['name'] == 'RandomForest한글'
        assert model['description'] == '랜덤 포레스트 분류기 🌲'
        assert model['author'] == 'ML팀'
        assert '기계학습' in model['tags']
        assert '🚀' in model['tags']
        assert 'This model supports 한글 and emojis 😊' == model['notes']


class TestCatalogParserIntegration:
    """카탈로그 파서 통합 테스트 (실제 사용 시나리오)"""
    
    def test_catalog_parser_cli_integration_scenario(self):
        """케이스 A: CLI에서 모델 목록 조회 시나리오"""
        # Given: CLI가 사용할 수 있는 완전한 카탈로그
        cli_catalog = {
            'classification': [
                {'name': 'RandomForest', 'module': 'sklearn.ensemble', 'class': 'RandomForestClassifier'},
                {'name': 'XGBoost', 'module': 'xgboost', 'class': 'XGBClassifier'},
                {'name': 'LogisticRegression', 'module': 'sklearn.linear_model', 'class': 'LogisticRegression'}
            ],
            'regression': [
                {'name': 'LinearRegression', 'module': 'sklearn.linear_model', 'class': 'LinearRegression'},
                {'name': 'RandomForestRegressor', 'module': 'sklearn.ensemble', 'class': 'RandomForestRegressor'}
            ],
            'clustering': [
                {'name': 'KMeans', 'module': 'sklearn.cluster', 'class': 'KMeans'},
                {'name': 'DBSCAN', 'module': 'sklearn.cluster', 'class': 'DBSCAN'}
            ]
        }
        
        cli_yaml_content = yaml.dump(cli_catalog)
        
        # When: CLI가 모델 카탈로그를 로드
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=cli_yaml_content):
            
            catalog = load_model_catalog()
        
        # Then: CLI가 필요한 모든 정보를 얻을 수 있음
        # 분류 모델 개수 확인
        classification_models = catalog.get('classification', [])
        assert len(classification_models) == 3
        
        # 회귀 모델 개수 확인
        regression_models = catalog.get('regression', [])
        assert len(regression_models) == 2
        
        # 클러스터링 모델 개수 확인
        clustering_models = catalog.get('clustering', [])
        assert len(clustering_models) == 2
        
        # 특정 모델 검색 시나리오
        xgboost_found = any(model['name'] == 'XGBoost' for model in classification_models)
        assert xgboost_found
        
        linear_regression_found = any(model['name'] == 'LinearRegression' for model in regression_models)
        assert linear_regression_found
    
    def test_catalog_parser_empty_categories_handling(self):
        """케이스 B: 일부 카테고리가 비어있는 카탈로그"""
        # Given: 일부 카테고리만 모델이 있는 카탈로그
        partial_catalog = {
            'classification': [
                {'name': 'RandomForest', 'module': 'sklearn.ensemble', 'class': 'RandomForestClassifier'}
            ],
            'regression': [],  # 빈 카테고리
            'clustering': [
                {'name': 'KMeans', 'module': 'sklearn.cluster', 'class': 'KMeans'}
            ],
            'timeseries': []  # 빈 카테고리
        }
        
        partial_yaml_content = yaml.dump(partial_catalog)
        
        # When: 부분적 카탈로그 로드
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=partial_yaml_content):
            
            catalog = load_model_catalog()
        
        # Then: 빈 카테고리도 올바르게 처리됨
        assert 'classification' in catalog
        assert 'regression' in catalog
        assert 'clustering' in catalog
        assert 'timeseries' in catalog
        
        assert len(catalog['classification']) == 1
        assert len(catalog['regression']) == 0  # 빈 리스트
        assert len(catalog['clustering']) == 1
        assert len(catalog['timeseries']) == 0  # 빈 리스트
    
    def test_catalog_parser_fallback_behavior(self):
        """케이스 C: 카탈로그 로드 실패 시 fallback 동작"""
        # Given: 다양한 실패 시나리오들
        failure_scenarios = [
            # 파일 없음
            (False, "", Exception("File not found")),
            # 파싱 에러
            (True, "invalid: yaml: content: [", yaml.YAMLError("Invalid YAML")),
            # 읽기 에러
            (True, "", IOError("Cannot read file"))
        ]
        
        for file_exists, file_content, mock_exception in failure_scenarios:
            # When: 각 실패 시나리오에서 카탈로그 로드
            with patch('pathlib.Path.exists', return_value=file_exists):
                if mock_exception:
                    with patch('pathlib.Path.read_text', side_effect=mock_exception):
                        result = load_model_catalog()
                else:
                    with patch('pathlib.Path.read_text', return_value=file_content):
                        result = load_model_catalog()
                
                # Then: 모든 실패 시나리오에서 빈 딕셔너리 반환 (CLI 중단 방지)
                assert result == {} or result is None
                
    def test_catalog_parser_performance_large_catalog(self):
        """케이스 D: 대용량 카탈로그 처리 성능"""
        # Given: 대용량 카탈로그 (100개 모델)
        large_catalog = {}
        
        for task_type in ['classification', 'regression', 'clustering', 'timeseries']:
            large_catalog[task_type] = []
            for i in range(25):  # 각 타입별 25개 모델
                model = {
                    'name': f'{task_type.title()}Model{i}',
                    'module': f'models.{task_type}',
                    'class': f'Model{i}Class',
                    'data_handler': task_type,
                    'hyperparameters': {
                        'param1': list(range(10)),
                        'param2': [f'value_{j}' for j in range(5)],
                        'param3': {'nested': {'deep': list(range(3))}}
                    },
                    'description': f'This is model {i} for {task_type} tasks. ' * 10  # 긴 설명
                }
                large_catalog[task_type].append(model)
        
        large_yaml_content = yaml.dump(large_catalog)
        
        # When: 대용량 카탈로그 로드
        import time
        start_time = time.time()
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=large_yaml_content):
            
            result = load_model_catalog()
        
        end_time = time.time()
        load_time = end_time - start_time
        
        # Then: 대용량 카탈로그도 빠르게 로드됨 (1초 이내)
        assert load_time < 1.0  # 성능 기준
        assert isinstance(result, dict)
        assert len(result) == 4  # 4개 카테고리
        
        # 각 카테고리별 25개 모델 확인
        for task_type in ['classification', 'regression', 'clustering', 'timeseries']:
            assert len(result[task_type]) == 25
            
        # 복잡한 중첩 구조가 올바르게 파싱되었는지 확인
        first_model = result['classification'][0]
        assert 'hyperparameters' in first_model
        assert 'param3' in first_model['hyperparameters']
        assert 'nested' in first_model['hyperparameters']['param3']
        assert 'deep' in first_model['hyperparameters']['param3']['nested']