// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rootcoord

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/internal/util/proxyutil"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"go.uber.org/zap"
)

type truncateCollectionTask struct {
	baseTask
	Req *milvuspb.TruncateCollectionRequest
}

// To be state
/**
undo.AddStep(CreateTempCollection, RemoveTempCollection) // 1
undo.AddStep(CreateIndexesForCollection, nullstep) // 2
undo.AddStep(LoadTempCollection, nullstep) // 3
undo.AddStep(ExchangeCollections, nullstep) // 4
redo.AddStep(RemoveOriginalCollection) // 5
*/

func (t *truncateCollectionTask) validate(ctx context.Context) error {
	if err := CheckMsgType(t.Req.GetBase().GetMsgType(), commonpb.MsgType_TruncateCollection); err != nil {
		return err
	}

	dbName := t.Req.GetDbName()
	collectionName := t.Req.GetCollectionName()
	if t.core.meta.IsAlias(ctx, dbName, collectionName) {
		return fmt.Errorf("can not truncate collection via alias")
	}

	return nil
}

func (t *truncateCollectionTask) Prepare(ctx context.Context) error {
	return t.validate(ctx)
}

func (t *truncateCollectionTask) Execute(ctx context.Context) error {
	dbName := t.Req.GetDbName()
	collName := t.Req.GetCollectionName()

	log.With(
		zap.String("dbName", dbName),
		zap.String("collection", collName),
	)

	// get latest collection from meta table by requesting with maxTS
	collMeta, err := t.core.meta.GetCollectionByName(ctx, dbName, collName, typeutil.MaxTimestamp)
	if err != nil {
		log.Error(fmt.Sprintf("failed to get collection by name, err: %+v", err))
		return err
	}

	tempCollName := util.GenerateTempCollectionName(collName)
	tempCollMeta, err := t.core.meta.GetCollectionByName(ctx, dbName, tempCollName, typeutil.MaxTimestamp)
	if err != nil {
		log.Error(fmt.Sprintf("failed to get temp collection, err: %+v", err))
		return err
	}

	undoTask := newBaseUndoTask(t.core.stepExecutor)
	// 1. Clean meta cache of all aliases of the original collection
	aliases := t.core.meta.ListAliasesByID(ctx, collMeta.CollectionID)
	ts := t.GetTs()
	undoTask.AddStep(
		&expireCacheStep{
			baseStep:        baseStep{core: t.core},
			dbName:          dbName,
			collectionNames: append(aliases, collMeta.Name),
			collectionID:    collMeta.CollectionID,
			ts:              ts,
			opts:            []proxyutil.ExpireCacheOpt{proxyutil.SetMsgType(commonpb.MsgType_TruncateCollection)},
		},
		&nullStep{},
	)
	// 2. build index for the temp collection
	undoTask.AddStep(
		&buildIndexStep{
			baseStep:     baseStep{core: t.core},
			collectionID: tempCollMeta.CollectionID,
		},
		&dropIndexStep{
			baseStep: baseStep{core: t.core},
			collID:   tempCollMeta.CollectionID,
		},
	)
	// 3. Release the original collection
	undoTask.AddStep(
		&releaseCollectionStep{},
		&nullStep{},
	)
	// 4. Load the temporary collection
	undoTask.AddStep(
		&loadCollectionStep{},
		&nullStep{},
	)
	// 5. Exchange original collection with the temporary one
	undoTask.AddStep(
		&exchangeCollectionStep{},
		&nullStep{},
	)
	// At this point , the original collection has been marked as DROP,
	// then eventually it will be removed by the async. background GC

	return undoTask.Execute(ctx)
}

func (t *truncateCollectionTask) GetLockerKey() LockerKey {
	return NewLockerKeyChain(
		NewClusterLockerKey(false),
		NewDatabaseLockerKey(t.Req.GetDbName(), false),
		NewCollectionLockerKey(t.Req.GetCollectionName(), true),
	)
}
