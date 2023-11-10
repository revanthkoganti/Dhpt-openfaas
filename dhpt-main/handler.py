import itertools
import numpy as np
import json
import asyncio
import aiohttp

async def make_request(session, function_url, payload):
    async with session.post(function_url, json=payload) as response:
        return await response.text()

async def handle(req):
    data = json.loads(req)
    subspaces = data.get('subspace')
    rf_params = {
        'max_depth': ['None'] + list(np.arange(10, 110, 10)),
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators': [100, 200, 300, 400, 500]
    }

    param_combinations = list(itertools.product(*rf_params.values()))

    subspaces = [{'max_depth': str(p[0]),
                 'max_features': p[1],
                 'min_samples_split': p[2],
                 'min_samples_leaf': p[3],
                 'n_estimators': p[4]} for p in param_combinations]

    total_combinations = len(subspaces)
    subspace1 = subspaces[:total_combinations // 2]
    subspace2 = subspaces[total_combinations // 2:]

    function1_url = "http://172.22.85.50:31112/function/grid-search"

    payload1 = {"subspace": subspace1}
    payload2 = {"subspace": subspace2}

    async with aiohttp.ClientSession() as session:
        response1, response2 = await asyncio.gather(
            make_request(session, function1_url, payload1),
            make_request(session, function1_url, payload2)
        )

    combined_response = {
        "response1": json.dumps(response1),
        "response2": json.dumps(response2)
    }

    return combined_response