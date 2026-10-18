from app.core.knowledge_review import is_pending


def test_extraction_resolution_is_closed_without_business_confirmation():
    assert not is_pending({'status': 'resolved_extraction', 'confirmed_by': None})
    assert not is_pending({'status': 'confirmed'})
    assert is_pending({'status': 'pending'})
    assert is_pending({})


def test_review_export_contains_only_pending(monkeypatch):
    import app.core.knowledge_review as module
    items = [dict(id='fixed', status='resolved_extraction', city='Москва', issue='fixed', file='a'),
             dict(id='pending', status='pending', city='Дубай', issue='period', file='b')]
    monkeypatch.setattr(module, 'load_queue', lambda: {'issues': items})
    exported = module.csv_export().decode('utf-8-sig')
    assert 'pending' in exported and 'fixed' not in exported
