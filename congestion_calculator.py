def calculate_weighted_score(counts):
    return (
        counts.get("car", 0) * 1 +
        counts.get("bike", 0) * 0.5 +
        counts.get("bus", 0) * 3 +
        counts.get("truck", 0) * 3
    )


def calculate_congestion(road_data):
    results = {}

    for road, data in road_data.items():
        counts = data["counts"]
        lanes = data["lanes"]

        weighted_score = calculate_weighted_score(counts)

        # Avoid division issues
        lanes = max(lanes, 1)

        congestion_score = weighted_score / lanes

        # Minimum baseline (important)
        congestion_score = max(congestion_score, 5)

        results[road] = {
            "score": round(congestion_score, 2),
            "lanes": lanes
        }
    return results


def get_priority_road(results):
    return max(results, key=lambda x: results[x]["score"])